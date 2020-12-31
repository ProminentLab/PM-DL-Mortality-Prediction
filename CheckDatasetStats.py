import pandas as pd
import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from datetime import datetime

from mimic3.database import MIMIC

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

def checkIcuAdmissions(db, hadm_id):
    query = "SELECT a.icustay_id FROM icustays a WHERE a.hadm_id=" + str(hadm_id) + " ORDER BY a.intime LIMIT 1;"
    icus = db.query(query)
    return len(icus)

def getPatient(db, subject_id):
    query = "SELECT p.subject_id, p.gender, p.dob, p.dod FROM patients p WHERE p.subject_id=" + str(subject_id) + ";"
    patient = db.query(query)
    return patient

def loadData(file, db, meta):
    y_pred = []
    y_true = []
    y_prob = []
    with open(file) as json_file:
        tss = json.load(json_file)
        for sample in tss:
            if sample["nextEvent"] is not None:
                hadm_id = meta[sample["patient"]]["admission_ids"][-1]
                prob = getAPSIII(db, hadm_id)
                y_prob.append(prob)

                if prob < 0:
                    continue
                else:
                    if prob >= 0.15:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)

                    if sample["nextEvent"] == "True":
                        y_true.append(1)
                    else:
                        y_true.append(0)
    return y_pred, y_true, y_prob


def checkICUAdmission(file, db, meta):
    filter = []
    cnt_false = 0
    total_admissions = 0
    with open(file) as json_file:
        tss = json.load(json_file)
        for sample in tss:
            if sample["nextEvent"] is not None:
                total_admissions +=1
                hadm_id = meta[sample["patient"]]["admission_ids"][-1]
                lens = checkIcuAdmissions(db,hadm_id)
                if lens < 1:
                    cnt_false += 1
                    filter.append(sample["patient"])
    print("no ICU admissions:", cnt_false)
    print("total admissions:", total_admissions)
    return filter

def difference(s, e):
    y = s.year - e.year
    m = s.month - e.month
    d = s.day - e.day
    if m < 0:
        y = y - 1
        m = m + 12
    if m == 0:
        if d < 0:
            m = m - 1
        elif d == 0:
            s1 = s.hour * 3600 + s.minute * 60 + s.second
            s2 = e.hour * 3600 + e.minut * 60 + e.second
            if s1 < s2:
                m = m - 1
    return '{}y{}m'.format(y, m)

def checkPatients(db, meta, patients):
    cnt = 0
    age_list = []
    male = 0
    died = 0
    admissions = []
    durations = []

    query = "SELECT p.subject_id, p.gender, p.dob, p.dod FROM patients p WHERE p.subject_id IN {}".format(
        tuple(patients))
    patient_data = db.query(query)


    for patient in patients:
        cnt += 1
        # Need to find patients gender
        # need to find patients age and if patient died
        met = meta[str(patient)]
        patient_details = patient_data[patient_data.subject_id == patient]
        if patient_details.gender.iloc[0] == "M":
            male += 1

        if met["in_hospital_death"][-1]:
            died += 1

        admissions.append(len(met["admission_ids"]))

        dob_year = patient_details.iloc[0]["dob"].year
        first_admit_year = datetime.fromtimestamp(int(met['admission_admittimes'][0])).year
        age = first_admit_year - dob_year
        if age >= 300:
            age = 91
        age_list.append(age)

        start_time = 0
        end_time = 0
        query = "SELECT * FROM admissions a WHERE a.subject_id=" + str(patient) + " ORDER BY a.admittime DESC LIMIT 1;"
        last_admission = db.query(query)
        for _, admission in last_admission.iterrows():
            end_time = admission.admittime
            break

        query = "SELECT * FROM admissions a WHERE a.subject_id=" + str(patient) + " ORDER BY a.admittime ASC LIMIT 1;"
        first_admission = db.query(query)
        for _, admission in first_admission.iterrows():
            start_time = admission.admittime
            break

        query = "SELECT c.charttime FROM labevents c WHERE c.subject_id=" + str(patient) + " AND c.itemid IN (50852, 50854, 50931, 50912) ORDER BY c.charttime ASC LIMIT 1;"
        labevents = db.query(query)
        for _, labevent in labevents.iterrows():
            if labevent.charttime < start_time:
                start_time = labevent.charttime
            break

        durations.append(len(pd.date_range(start=start_time,end=end_time,freq='D'))/365)

    return cnt, male/cnt, np.mean(age_list), 1-(died/cnt), np.mean(admissions), np.mean(durations)

if __name__== "__main__":
    db = MIMIC()

    df1 = pd.read_csv("data/patients_train.csv")
    df2 = pd.read_csv("data/patients_val.csv")
    filter = pd.read_csv("data/patients_negative_examples.csv")
    filter = filter.subject_id.tolist()

    df = pd.concat([df1, df2])
    df = df.subject_id.tolist()

    train_patients = []
    for patient in df:
        if patient not in filter:
            train_patients.append(patient)

    df = pd.read_csv("data/patients_test.csv")
    test_patients = df.subject_id.tolist()

    with open('data/output/meta.json') as json_file:
        meta = json.load(json_file)

    print("TRAIN:")
    num_train_patients, males, avg_age, died, admissions, durations = checkPatients(db, meta, train_patients)

    print("Num patients:", num_train_patients)
    print("Males:", males)
    print("Average age:", avg_age)
    print("Died:", died)
    print("Admission:", admissions)
    print("Durations:", durations)

    print()
    print("TEST:")
    num_test_patients, males, avg_age, died, admissions, durations = checkPatients(db, meta, test_patients)
    print("Num patients:", num_test_patients)
    print("Males:", males)
    print("Average age:", avg_age)
    print("Died:", died)
    print("Admission:", admissions)
    print("Durations:", durations)




