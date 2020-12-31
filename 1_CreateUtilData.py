import pandas as pd

from mimic3.database import MIMIC
from icd.icd9to10 import Icd9to10
from hcuppy.elixhauser import ElixhauserEngine

if __name__ == "__main__":
    db = MIMIC()

    print("*** Get Comorbidity Information ***")
    icd9to10 = Icd9to10()
    df = pd.read_csv("data/patients_train.csv")
    df1 = pd.read_csv("data/patients_val.csv")
    df2 = pd.read_csv("data/patients_test.csv")
    df = pd.concat([df, df1, df2])
    df.drop_duplicates(subset=None, keep='first', inplace=True)

    lines9 = []
    lines10 = []
    for _, row in df.iterrows():
        subject_id = row.subject_id

        query = "SELECT a.subject_id, a.hadm_id, a.dischtime, a.admittime, a.insurance, a.admission_type, a.deathtime, a.ethnicity FROM admissions a WHERE a.subject_id=" + str(
            subject_id) + " ORDER BY a.dischtime ASC;"
        admissions = db.query(query)

        for _, admission in admissions.iterrows():
            hadm_id = admission["hadm_id"]

            out9 = str(hadm_id)
            out10 = str(hadm_id)
            query = "SELECT d.subject_id, d.hadm_id, d.seq_num, d.icd9_code FROM diagnoses_icd d WHERE d.hadm_id=" + str(
                hadm_id) + " ORDER BY d.seq_num ASC;"
            results = db.query(query)
            for _, code in results.iterrows():
                out9 += "," + str(code.icd9_code)

                icd10s = icd9to10.icd9to10(code.icd9_code)
                for icd10 in icd10s:
                    out10 += "," + str(icd10)

            lines9.append(out9)
            lines10.append(out10)

    with open('data/util/admission_icd9_sequences.txt', 'a') as the_file:
        for line in lines9:
            the_file.write(line + '\n')

    with open('data/util/admission_icd10_sequences.txt', 'a') as the_file:
        for line in lines10:
            the_file.write(line + '\n')

    print("*** Write Elixhauser Comorbidity Sequences ***")
    ee = ElixhauserEngine()
    elix_lines = []
    with open('data/util/admission_icd10_sequences.txt') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line.replace("\n", "")
            array = line.split(",")
            hadm_id = array[0]

            elix_line = hadm_id

            icds = list(array[1:])

            elix = ee.get_elixhauser(icds)
            elix_line += "," + str(elix["rdmsn_scr"]) + "," + str(elix["mrtlt_scr"])

            for comorbidity in elix["cmrbdt_lst"]:
                elix_line += "," + comorbidity

            elix_lines.append(elix_line)

    with open('data/util/admission_elix_sequences.txt', 'a') as the_file:
        for line in elix_lines:
            the_file.write(line + '\n')


    print("*** Load Severity Scores ***")

    oasis_lines = []
    sofa_lines = []
    sapsii_lines = []
    apsiii_lines = []
    for _, row in df.iterrows():
        subject_id = row.subject_id

        query = "SELECT a.subject_id, a.hadm_id FROM admissions a WHERE a.subject_id=" + str(subject_id) + " ORDER BY a.dischtime ASC;"
        admissions = db.query(query)

        for _, admission in admissions.iterrows():
            hadm_id = admission["hadm_id"]

            query = "SELECT a.icustay_id FROM icustays a WHERE a.hadm_id=" + str(
                    hadm_id) + " ORDER BY a.intime LIMIT 1;"
            icus = db.query(query)

            icustay_id = None
            for _, icu in icus.iterrows():
                icustay_id = icu["icustay_id"]

                query = "SELECT d.hadm_id, d.oasis, d.age_score, d.gcs_score, d.heartrate_score, d.meanbp_score, d.resprate_score, d.temp_score, d.urineoutput_score, d.mechvent_score, d.electivesurgery_score FROM oasis d WHERE d.icustay_id=" + str(
                    icustay_id) + ";"
                results = db.query(query)
                for _, row in results.iterrows():
                    out = str(hadm_id) + "," + str(row.oasis) + "," + str(row.age_score) + "," + str(
                            row.gcs_score) + "," + str(row.heartrate_score) + "," + str(row.meanbp_score) + "," + str(
                            row.resprate_score) + "," + str(row.temp_score) + "," + str(
                            row.urineoutput_score) + "," + str(row.mechvent_score) + "," + str(
                            row.electivesurgery_score)
                    out = out.replace("None", "-1")
                    out = out.replace("nan", "-1")
                oasis_lines.append(out)

                query = "SELECT * FROM sofa d WHERE d.icustay_id=" + str(icustay_id) + ";"
                results = db.query(query)
                for _, row in results.iterrows():
                    out = str(hadm_id) + "," + str(row.sofa) + "," + str(row.respiration) + "," + str(
                            row.coagulation) + "," + str(row.liver) + "," + str(row.cardiovascular) + "," + str(
                            row.cns) + "," + str(row.renal)
                    out = out.replace("None", "-1")
                    out = out.replace("nan", "-1")
                sofa_lines.append(out)

                query = "SELECT * FROM sapsii d WHERE d.icustay_id=" + str(icustay_id) + ";"
                results = db.query(query)
                for _, row in results.iterrows():
                    out = str(hadm_id) + "," + str(row.sapsii) + "," + str(row.age_score) + "," + str(
                            row.hr_score) + "," + str(row.sysbp_score) + "," + str(row.temp_score) + "," + str(
                            row.pao2fio2_score) + "," + str(row.uo_score) + "," + str(row.bun_score) + "," + str(
                            row.wbc_score) + "," + str(row.potassium_score) + "," + str(row.sodium_score) + "," + str(
                            row.bicarbonate_score) + "," + str(row.bilirubin_score) + "," + str(
                            row.gcs_score) + "," + str(row.comorbidity_score) + "," + str(row.admissiontype_score)
                    out = out.replace("None", "-1")
                    out = out.replace("nan", "-1")
                sapsii_lines.append(out)

                query = "SELECT * FROM apsiii d WHERE d.icustay_id=" + str(icustay_id) + ";"
                results = db.query(query)
                for _, row in results.iterrows():
                    out = str(hadm_id) + "," + str(row.apsiii) + "," + str(row.hr_score) + "," + str(
                            row.meanbp_score) + "," + str(row.temp_score) + "," + str(row.resprate_score) + "," + str(
                            row.hematocrit_score) + "," + str(row.wbc_score) + "," + str(
                            row.creatinine_score) + "," + str(row.uo_score) + "," + str(row.bun_score) + "," + str(
                            row.glucose_score) + "," + str(row.acidbase_score) + "," + str(row.gcs_score)
                    out = out.replace("None", "-1")
                    out = out.replace("nan", "-1")
                apsiii_lines.append(out)

    with open('data/util/admission_oasis.txt', 'a') as the_file:
        for line in oasis_lines:
            the_file.write(line + '\n')

    with open('data/util/admission_sofa.txt', 'a') as the_file:
        for line in sofa_lines:
            the_file.write(line + '\n')

    with open('data/util/admission_sapsii.txt', 'a') as the_file:
        for line in sapsii_lines:
            the_file.write(line + '\n')

    with open('data/util/admission_apsiii.txt', 'a') as the_file:
        for line in apsiii_lines:
            the_file.write(line + '\n')