# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961793/#r12-2838534
from mimic3.database import MIMIC
import numpy as np
import pandas as pd
import json


def getAdmissions(db, patient):
    query = "SELECT a.subject_id, a.hadm_id, a.dischtime, a.admittime, a.insurance, a.admission_type, a.deathtime, a.ethnicity, a.admission_location FROM admissions a WHERE a.subject_id=" + str(
        patient) + " ORDER BY a.dischtime ASC;"
    admissions = db.query(query)
    return admissions


def getLabEvents(db, admission_id):
    query = "SELECT c.subject_id, c.hadm_id, c.itemid, c.charttime, c.value, c.valuenum, c.valueuom, c.flag FROM labevents c WHERE c.hadm_id=" + str(
        admission_id) + " AND c.itemid IN (50852, 50854, 50931, 50912) ORDER BY c.charttime ASC;"
    labevents = db.query(query)
    return labevents

def encode_insurance(insurance):
    if "government" in str(insurance).lower():
        val = 0
    elif "self " in str(insurance).lower():
        val = 1
    elif "medicare" in str(insurance).lower():
        val = 2
    elif "private" in str(insurance).lower():
        val = 3
    elif "medicaid" in str(insurance).lower():
        val = 4

    return insurance
    #return val

def encode_admissiontype(admission_type):
    val = -1
    if admission_type == "URGENT" or admission_type == "EMERGENCY":
        val = 1
    elif admission_type == "NEWBORN" or admission_type == "ELECTIVE":
        val = 0
    return admission_type

def encode_ethnicity(ethnicity):
    if "white" in str(ethnicity).lower():
        val = 1
    elif "black" in str(ethnicity).lower():
        val = 2
    elif "asian" in str(ethnicity).lower() or "middle eastern" in str(ethnicity).lower():
        val = 3
    elif "latino" in str(ethnicity).lower() or "hispanic" in str(ethnicity).lower():
        val = 4
    else:
        val = 0
    return val

def encodeAdmissionLocation(location):
    """
    if location == "CLINIC REFERRAL/PREMATURE":
        return 1
    elif location == "TRANSFER FROM HOSP/EXTRAM":
        return 3
    elif location == "EMERGENCY ROOM ADMIT":
        return 4
    elif location == "TRANSFER FROM SKILLED NUR":
        return 5
    elif location == "PHYS REFERRAL/NORMAL DELI":
        return 6
    elif location == "TRANSFER FROM OTHER HEALT":
        return 7
    elif location == "HMO REFERALL/SICK":
        return 8
    elif location == "TRST WITHIN THIS FACILITY":
        return 9
    else:
        return 2
    """
    return location



def getDemographics(db, hadm_id):
    query = "SELECT c.age, c.gender FROM icustay_detail c WHERE c.hadm_id=" + str(hadm_id) + ";"
    results = db.query(query)
    gender = 0
    age = 0
    agebracket = 0
    for _, row in results.iterrows():
        gender = row.gender
        #if gender == "M":
        #    gender = 1
        #else:
        #    gender = 0

        age = row.age

        if age > 80:
            agebracket = 5
        elif age > 70:
            agebracket = 4
        elif age > 60:
            agebracket = 3
        elif age > 50:
            agebracket = 2
        else:
            agebracket = 1

    return gender, age, agebracket

def getMeanBloodGlucose(db, hadm_id):
    query = "SELECT c.subject_id, c.hadm_id, c.itemid, c.charttime, c.value, c.valuenum, c.valueuom, c.flag FROM labevents c WHERE c.hadm_id=" + str(hadm_id) + \
            " AND c.itemid IN (50931) ORDER BY c.charttime ASC;"
    labevents = db.query(query)

    vals = []
    for _, lab in labevents.iterrows():
        vals.append(lab.valuenum)

    mean = np.mean(vals)
    if np.isnan(mean):
        return 0
    else:
        return mean

def getHb1acs(db, hadm_id):
    query = "SELECT c.subject_id, c.hadm_id, c.itemid, c.charttime, c.value, c.valuenum, c.valueuom, c.flag FROM labevents c WHERE c.hadm_id=" + str(hadm_id) + \
            " AND c.itemid IN (50852, 50854) ORDER BY c.charttime ASC;"
    labevents = db.query(query)
    vals = []
    for _, lab in labevents.iterrows():
        vals.append(lab.valuenum)
    return vals

def getElixhauser(db, hadm_id):
    query = "SELECT c.elixhauser_vanwalraven FROM elixhauser_ahrq_score c WHERE c.hadm_id=" + str(hadm_id) + ";"
    elixs = db.query(query)
    for _, elix in elixs.iterrows():
        elix = elix.elixhauser_vanwalraven
        break
    return elix

def getSerumScore(db, hadm_id):
    query = "SELECT c.valuenum FROM labevents c WHERE c.hadm_id=" + str(hadm_id) + " AND c.itemid IN (50912) ORDER BY c.charttime ASC;"
    labevents = db.query(query)
    score = 0
    for _, lab in labevents.iterrows():
        if lab.valuenum > 2.5 and score < 2:
            score = 2
        elif lab.valuenum > 1.5 and score < 1:
            score = 1
    return score

def getDiagnoses(db, hadm_id):
    query = "SELECT d.subject_id, d.hadm_id, d.seq_num, d.icd9_code FROM diagnoses_icd d WHERE d.hadm_id=" + str(hadm_id) + " ORDER BY d.seq_num ASC;"
    icds = db.query(query)
    diagnose_codes = []
    for _, icd in icds.iterrows():
        diagnose_codes.append(icd.icd9_code)
    return diagnose_codes

def getEthnicity(db, hadm_id):
    query = "SELECT d.ethnicity FROM icustay_detail d WHERE d.hadm_id=" + str(hadm_id) + ";"
    data = db.query(query)
    for _, row in data.iterrows():
        return row.ethnicity
        break


def ophtalmic(icd):
    score = 0

    if str(icd).startswith("2505"):
        score = 1
    if str(icd).startswith("2495"):
        score = 1
    if str(icd).startswith("3620") and str(icd) != "36202":
        score = 1
    if str(icd).startswith("3621"):
        score = 1
    if str(icd) == "36253":
        score = 1
    if str(icd).startswith("36281"):
        score = 1
    if str(icd).startswith("36282"):
        score = 1
    if str(icd).startswith("36283"):
        score = 1
    if str(icd).startswith("361"):
        score = 2
    if str(icd) == "36202":
        score = 2
    if str(icd).startswith("369"):
        score = 2
    if str(icd) == "37923":
        score = 2

    return score

def nephropathy(icd):
    score = 0

    if str(icd).startswith("2504"):
        score = 1
    if str(icd).startswith("580"):
        score = 1
    if str(icd).startswith("581"):
        score = 1
    if str(icd).startswith("582"):
        score = 1
    if str(icd).startswith("583"):
        score = 1
    if str(icd) == "5851":
        score = 1
    if str(icd) == "5852":
        score = 1
    if str(icd) == "5853":
        score = 1
    if str(icd) == "5859":
        score = 1
    if str(icd).startswith("5854"):
        score = 2
    if str(icd).startswith("5855"):
        score = 2
    if str(icd).startswith("5856"):
        score = 2
    if str(icd) == "586":
        score = 2
    if str(icd) == "5939":
        score = 2

    return score

def neuropathy(icd):
    score = 0

    if str(icd).startswith("2506"):
        score = 1
    if str(icd) == "3572":
        score = 1
    if str(icd).startswith("2496"):
        score = 1
    if str(icd).startswith("3370"):
        score = 1
    if str(icd) == "3371":
        score = 1
    if str(icd).startswith("354"):
        score = 1
    if str(icd).startswith("355"):
        score = 1
    if str(icd) == "3569":
        score = 1
    if str(icd) == "3581":
        score = 1
    if str(icd) == "4580":
        score = 1
    if str(icd) == "5363":
        score = 1
    if str(icd) == "5645":
        score = 1
    if str(icd) == "59654":
        score = 1
    if str(icd) == "7135":
        score = 1
    if str(icd) == "9510":
        score = 1
    if str(icd) == "9511":
        score = 1
    if str(icd) == "9513":
        score = 1

    return score

def cerebrovascular(icd):
    score = 0

    if str(icd).startswith("435"):
        score = 1
    if str(icd) == "431" or str(icd) == "436" or str(icd).startswith("433") or str(icd).startswith("434"):
        score = 2

    return score

def cardiovascular(icd):
    score = 0

    if str(icd).startswith("411"):
        score = 1
    if str(icd).startswith("413"):
        score = 1
    if str(icd).startswith("414"):
        score = 1
    if str(icd) == "4292":
        score = 1
    if str(icd) != "44023" and str(icd) != "44024" and str(icd).startswith("440"):
        score = 1
    if str(icd).startswith("410"):
        score = 2
    if str(icd) == "412":
        score = 2
    if str(icd).startswith("4273"):
        score = 2
    if str(icd) == "4275":
        score = 2
    if str(icd) == "4271":
        score = 2
    if str(icd).startswith("4274"):
        score = 2
    if str(icd).startswith("428"):
        score = 2
    if str(icd) == "44023" or str(icd) == "44024":
        score = 2
    if str(icd).startswith("441"):
        score = 2

    return score

def peripheral(icd):
    score = 0

    if str(icd).startswith("2507"):
        score = 1
    if str(icd).startswith("2497"):
        score = 1
    if str(icd) == "4423":
        score = 1
    if str(icd) == "44021" or str(icd) == "44381" or str(icd) == "4439":
        score = 1
    if str(icd) == "8921":
        score = 1
    if str(icd) == "0400":
        score = 2
    if str(icd) == "44422":
        score = 2
    if str(icd).startswith("7071"):
        score = 2
    if str(icd) == "7854":
        score = 2

    return score

def metabolic(icd):
    score = 0

    if str(icd) == "2501":
        score = 2
    if str(icd).startswith("2501"):
        score = 2
    if str(icd).startswith("2491"):
        score = 2
    if str(icd).startswith("2502"):
        score = 2
    if str(icd).startswith("2492"):
        score = 2
    if str(icd).startswith("2503"):
        score = 2
    if str(icd).startswith("2493"):
        score = 2
    if str(icd).startswith("2508") or str(icd).startswith("2509"):
        score = 2
    if str(icd).startswith("2498") or str(icd).startswith("2499"):
        score = 2

    return score

def getDCSI(db, hadm_id):
    diagnoses = getDiagnoses(db, admission.hadm_id)
    ophtalmics = [0]
    nephropathys = [0]
    neuropathys = [0]
    cerebrovasculars = [0]
    cardiovasculars = [0]
    peripherals = [0]
    metabolics = [0]
    for icd in diagnoses:
        ophtalmics.append(ophtalmic(icd))
        nephropathys.append(nephropathy(icd))
        neuropathys.append(neuropathy(icd))
        cerebrovasculars.append(cerebrovascular(icd))
        cardiovasculars.append(cardiovascular(icd))
        peripherals.append(peripheral(icd))
        metabolics.append(metabolic(icd))
    nephropathys.append(getSerumScore(db, hadm_id))
    score = np.max(ophtalmics) + \
            np.max(nephropathys) + \
            np.max(neuropathys) + \
            np.max(cerebrovasculars) + \
            np.max(cardiovasculars) + \
            np.max(peripherals) + \
            np.max(metabolics)
    return score

def loadLabels(meta):
    df1 = pd.read_csv("../../data/patients_truely_icu.csv")
    df2 = pd.read_csv("../../data/patients_test.csv")
    df = pd.concat([df1, df2])
    patients = list(df.subject_id.unique())

    labels = dict()
    for patient in patients:
        if meta[str(patient)]["in_hospital_death"][-1] == True:
             labels[str(patient)] = 1
        else:
            labels[str(patient)] = 0

    return labels

if __name__== "__main__":
    charlson = pd.read_csv("patients_charlson_last.csv")
    #elixhauser = pd.read_csv("D:/mimic_data/patients_elix_last.csv")

    db = MIMIC()

    with open('../../data/output/meta.json') as json_file:
        meta = json.load(json_file)

    labels = loadLabels(meta)

    for mode in ["train", "test"]:
        dataframes = []

        if mode == "train":
            df1 = pd.read_csv("../../data/patients_train.csv")
            df2 = pd.read_csv("../../data/patients_val.csv")
            filter = pd.read_csv("../../data/patients_truely_icu.csv")
            filter = filter.subject_id.tolist()

            df = pd.concat([df1, df2])
            df = df.subject_id.tolist()

            patients = []
            for patient in df:
                if patient in filter:
                    patients.append(str(patient))

            print("Train patients", len(patients))

        else:
            df = pd.read_csv("../../data/patients_test.csv")
            df = df.subject_id.tolist()

            filter = pd.read_csv("../../data/patients_truely_icu.csv")
            filter = filter.subject_id.tolist()

            patients = []
            for patient in df:
                patients.append(str(patient))

            print("Test patients", len(patients))

        for patient in patients:
            dataframe = dict()

            admissions = getAdmissions(db, patient)
            hb1acs = []
            for i, (_, admission) in enumerate(admissions.iterrows()):
                if i == len(admissions) - 1:
                    admission_oi = admission
                hb1acs += getHb1acs(db, admission.hadm_id)

            mean_hb1ac = 0.0
            if len(hb1acs) > 0:
                mean_hb1ac = np.mean(hb1acs)

                if np.isnan(mean_hb1ac):
                    mean_hb1ac = 0.0

            dataframe["patient"] = patient
            dataframe["admission_oi"] = str(admission_oi.hadm_id)
            #dataframe["ethnicity"] = encode_ethnicity(admission_oi.ethnicity)
            dataframe["ethnicity"] = getEthnicity(db, admission_oi.hadm_id)
            dataframe["admission_type"] = encode_admissiontype(admission_oi.admission_type)
            dataframe["gender"], dataframe["age"], dataframe["age_bracket"] = getDemographics(db, admission_oi.hadm_id)
            dataframe["insurance"] = encode_insurance(admission_oi.insurance)
            dataframe["admission_location"] = encodeAdmissionLocation(str(admission_oi.admission_location))
            dataframe["mean_blood_glucose"] = getMeanBloodGlucose(db, admission_oi.hadm_id)
            dataframe["mean_hb1ac"] = mean_hb1ac
            dataframe["elixhauser"] = getElixhauser(db, admission_oi.hadm_id)
            #dataframe["elixhauser"] = elixhauser.loc[elixhauser['patient'] == int(patient)].iloc[0]["score"]
            dataframe["cci"] = charlson.loc[charlson['patient'] == int(patient)].iloc[0]["score"]
            dataframe["dcsi"] = getDCSI(db, admission_oi.hadm_id)
            dataframe["label"] = labels[patient]

            dataframes.append(dataframe)

        ### Convert to Dataframe
        rows = dict()
        columns = []
        for i, dataframe in enumerate(dataframes):
            vals = []
            for key in dataframe.keys():
                vals.append(dataframe[key])
                if i == 0:
                    columns.append(key)
            rows[i] = vals

        df = pd.DataFrame.from_dict(rows, orient='index', columns=columns)
        print(df)
        df.to_csv("../../data/util/ml_data_" + mode + ".csv", index=False)