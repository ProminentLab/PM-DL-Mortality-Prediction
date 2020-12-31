import json
from mimic3.database import MIMIC
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def getAPSIII(db, hadm_id):
    score = -1
    query = "SELECT a.icustay_id FROM icustays a WHERE a.hadm_id=" + str(hadm_id) + " ORDER BY a.intime LIMIT 1;"
    icus = db.query(query)
    for _, icu in icus.iterrows():
        score = 0
        icustay_id = icu["icustay_id"]
        query = "SELECT c.apsiii_prob FROM apsiii c WHERE c.icustay_id=" + str(icustay_id) + ";"
        rows = db.query(query)
        for _, row in rows.iterrows():
            score = row.apsiii_prob
            break
    return score

def loadData(file, db, meta, p_thresh):
    y_pred = []
    y_true = []
    y_prob = []
    y_score = []
    with open(file) as json_file:
        tss = json.load(json_file)
        for sample in tss:
            hadm_id = meta[sample["patient"]]["admission_ids"][-1]
            prob = getAPSIII(db, hadm_id)
            y_prob.append([1-prob, prob])
            y_score.append(prob)

            if prob < -100:
                print("TOO SMALL")
                continue
            else:
                if prob >= p_thresh:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

                if sample["nextEvent"] == "True":
                    y_true.append(1)
                else:
                    y_true.append(0)
    return y_pred, y_true, y_prob, y_score

def determine_optimal_thresh(meta):
    df1 = pd.read_csv("../data/patients_train.csv")
    df2 = pd.read_csv("../data/patients_val.csv")
    filter = pd.read_csv("../data/patients_truely_icu.csv")
    filter = filter.subject_id.tolist()

    df = pd.concat([df1, df2])
    df = df.subject_id.tolist()

    train_patients = []
    for patient in df:
        if patient not in filter:
            train_patients.append(patient)

    y_true = []
    y_prob = []
    y_score = []
    for patient in train_patients:
        hadm_id = meta[str(patient)]["admission_ids"][-1]
        prob = getAPSIII(db, hadm_id)
        y_prob.append([1 - prob, prob])
        y_score.append(prob)

        if prob < -100:
            print("TOO SMALL")
            continue
        else:
            if meta[str(patient)]["in_hospital_death"][-1] == True:
                y_true.append(1)
            else:
                y_true.append(0)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Train AUC:", roc_auc_score(y_true, y_score))
    print("Optimal threshold:", optimal_threshold)

    return optimal_threshold

if __name__== "__main__":
    db = MIMIC()
    with open('../data/output/meta.json') as json_file:
        meta = json.load(json_file)

    p_thresh = determine_optimal_thresh(meta)

    y_pred, y_true, y_prob, y_score = loadData("../data/output/sm_log_train_tss_test.json", db, meta, p_thresh=p_thresh)

    result_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 'y_score': y_prob, 'y_probscore':y_score})
    result_df.to_csv("../data/results/apsiii_results.csv", index=None)

    print("*** DONE ***")