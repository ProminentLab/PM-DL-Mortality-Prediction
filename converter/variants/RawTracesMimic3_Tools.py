import pandas as pd
import numpy as np

ranges_glucose = [100, 150, 200, 250, 300]
ranges_creatinine = [1.25, 2.5, 4, 5.5, 7]
ranges_hemo = [5, 6, 8, 10, 12, 14]

ranges_glucose_std = [50, 100, 150]
ranges_creatinine_std = [0.5, 1, 1.5, 2]
ranges_hemo_std = [0.1, 0.4, 0.8]

def discretize(value, list_values):
    index = len(list_values)
    for i in range(len(list_values)):
        if value < list_values[i]:
            index = i
            break
    return index

def findTimeRange(timestamp, start_times):
    time_index = len(start_times)
    for i in range(len(start_times)):
        if timestamp < start_times[i]:
            time_index = i
            break
    return time_index

class RawTracesMimic3_Tools:
    def __init__(self):
        pass

    def build(self, db):
        self.db = db

        self.comorbidities = self.loadComorbidities()

        self.rawTraces, self.rawMeta = self.getRawTraces()
        return self.rawTraces, self.rawMeta

    def loadComorbidities(self):
        lookup = dict()
        with open('data/util/admission_elix_sequences.txt') as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                line = line.replace("\n", "")
                array = line.split(",")
                hadm_id = array[0]
                comorbidity = list(array[3:])
                lookup[hadm_id] = comorbidity
        return lookup


    def getAPSIII(db, hadm_id):
        query = "SELECT c.apsiii_prob FROM apsiii c WHERE c.hadm_id=" + str(hadm_id) + ";"
        rows = db.query(query)
        score = 0
        for _, row in rows.iterrows():
            score = row.apsiii_prob
            break
        return score

    def getRawTraces(self):
        metadata = {}
        traces = list()
        patients = self.getDiabetesPatients()

        d_labitems = self.getDLabitems()

        trace_cnt = 0
        for index, patient in patients.iterrows():
            trace = {}
            trace["patient"] = patient["subject_id"]
            trace["gender"] = patient["gender"]
            trace["dob"] = patient["dob"].timestamp()
            trace["dod"] = str(patient["dod"])
            trace["events"] = list()

            dob_year = patient["dob"].year

            """ Get Admissions """
            admissions = self.getAdmissions(patient=trace["patient"])
            if len(admissions) < 2:
                trace_cnt += 1
                continue

            """ Check if first admission is NON-DIABETIC """
            for _, admission in admissions.iterrows():
                first_admit_year = admission.admittime.year
                trace["ethnicity"] = admission.ethnicity
                break
            """ --------------------------------> """

            """ Create in-patient and outpatient time intervals """
            admit_times = []
            for _, admission in admissions.iterrows():
                admit_times.append(admission.admittime.timestamp())
            """ --------------------------------> """

            trace["age"] = first_admit_year - dob_year
            if trace["age"] >= 300:
                trace["age"] = 91

            """ METADATA """
            meta = {}
            meta["patient"] = patient["subject_id"]
            meta["admission_ids"] = self.getAdmissionsList(admissions)
            meta["admission_admittimes"], meta["admission_dischargetimes"] = self.getAdmissionTimes(admissions)
            meta["admission_discharge_delta"] = self.getDeltasBetweenDischarges(admissions)
            meta["admission_admit_delta"] = self.getDeltasBetweenAdmissions(admissions)
            meta["admission_discharge_admit_delta"] = self.getDeltasBetweenDischargeAdmissions(admissions)
            meta["in_hospital_death"] = self.inHospitalDeaths(admissions)
            meta["diabetic_icd_index"] = list()
            meta["icd_indices"] = list()
            """ --------------------------------> """

            """ OUTPATIENT DATA """
            labevents = self.getLabEventsUnassigned(patient["subject_id"])
            time_intervals = {}
            for _, lab in labevents.iterrows():
                time_index = findTimeRange(lab["charttime"].timestamp(), admit_times)
                if time_index not in time_intervals.keys():
                    time_intervals[time_index] = []

                items = d_labitems.loc[d_labitems['itemid'] == lab.itemid]
                item = items.iloc[0]
                event = {}
                event["type"] = "lab_unassigned"
                event["timestamp"] = lab["charttime"].timestamp()
                event["label"] = item.label
                event["valuenum"] = lab.valuenum
                time_intervals[time_index].append(event)

            """ Process Outpatient Dictionary"""
            for key in time_intervals.keys():
                series = time_intervals[key]

                glucose_time_series = []
                glucose_first = None
                glucose_last = None

                serum_time_series = []
                serum_first = None
                serum_last = None

                hemo_time_series = []
                hemo_first = None
                hemo_last = None

                for event in series:
                    if event["label"] == "Glucose" and isinstance(event['valuenum'], (int, float)):
                        glucose_time_series.append(event['valuenum'])
                        if glucose_first is None:
                            glucose_first = event["timestamp"]
                            glucose_last = event["timestamp"]
                        if event["timestamp"] < glucose_first:
                            glucose_first = event["timestamp"]
                        if event["timestamp"] > glucose_last:
                            glucose_last = event["timestamp"]

                    if event["label"] == "Creatinine" and isinstance(event['valuenum'], (int, float)):
                        serum_time_series.append(event['valuenum'])
                        if serum_first is None:
                            serum_first = event["timestamp"]
                            serum_last = event["timestamp"]
                        if event["timestamp"] < serum_first:
                            serum_first = event["timestamp"]
                        if event["timestamp"] > serum_last:
                            serum_last = event["timestamp"]

                        if "A1c" in event["label"] and isinstance(event['valuenum'], (int, float)):
                            hemo_time_series.append(event['valuenum'])
                            if hemo_first is None:
                                hemo_first = event["timestamp"]
                                hemo_last = event["timestamp"]
                            if event["timestamp"] < hemo_first:
                                hemo_first = event["timestamp"]
                            if event["timestamp"] > hemo_last:
                                hemo_last = event["timestamp"]

                if len(glucose_time_series) > 0:
                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Glucose"
                    event["desc"] = "mean"
                    event["value"] = str(discretize(np.mean(glucose_time_series), ranges_glucose))
                    event["timestamp"] = glucose_first
                    trace["events"].append(event)

                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Glucose"
                    event["desc"] = "std"
                    event["value"] = str(discretize(np.std(glucose_time_series), ranges_glucose_std))
                    event["timestamp"] = glucose_last
                    trace["events"].append(event)

                if len(serum_time_series) > 0:
                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Creatinine"
                    event["desc"] = "mean"
                    event["value"] = str(discretize(np.mean(serum_time_series), ranges_creatinine))
                    event["timestamp"] = serum_first
                    trace["events"].append(event)

                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Creatinine"
                    event["desc"] = "std"
                    event["value"] = str(discretize(np.std(serum_time_series), ranges_creatinine_std))
                    event["timestamp"] = serum_last
                    trace["events"].append(event)

                if len(hemo_time_series) > 0:
                    event = {}
                    event["type"] = "lab"
                    event["label"] = "A1c"
                    event["desc"] = "mean"
                    event["value"] = str(discretize(np.mean(hemo_time_series), ranges_hemo))
                    event["timestamp"] = hemo_first
                    trace["events"].append(event)

                    event = {}
                    event["type"] = "lab"
                    event["label"] = "A1c"
                    event["desc"] = "std"
                    event["value"] = str(discretize(np.std(hemo_time_series), ranges_hemo_std))
                    event["timestamp"] = hemo_last
                    trace["events"].append(event)

            admission_cnt = -1
            isLastAdmission = False
            for _, admission in admissions.iterrows():
                admission_cnt += 1

                if len(admissions) == admission_cnt + 1:
                    isLastAdmission = True

                admission_id = admission["hadm_id"]
                dischtime = admission["dischtime"].timestamp()

                admittime = admission["admittime"].timestamp()
                admittime += 10

                event = {}
                event["type"] = "admission"
                event["timestamp"] = admission["admittime"].timestamp()
                event["value"] = admission["admission_type"]
                event["admission_id"] = admission_id
                event["deathtime"] = str(admission["deathtime"])
                trace["events"].append(event)

                event = {}
                event["type"] = "admission"
                event["timestamp"] = admission["admittime"].timestamp()
                event["value"] = admission["insurance"]
                event["admission_id"] = admission_id
                trace["events"].append(event)


                """ Process Lab Events """
                labevents = self.getLabEvents(admission_id)

                glucose_time_series = []
                glucose_first = None
                glucose_last = None

                serum_time_series = []
                serum_first = None
                serum_last = None

                hemo_time_series = []
                hemo_first = None
                hemo_last = None

                for _, lab in labevents.iterrows():
                    items = d_labitems.loc[d_labitems['itemid'] == lab.itemid]
                    item = items.iloc[0]
                    if item.label == "Glucose":
                        glucose_time_series.append(lab.valuenum)
                        if glucose_first is None:
                            glucose_first = lab["charttime"].timestamp()
                            glucose_last = lab["charttime"].timestamp()
                        if lab["charttime"].timestamp() < glucose_first:
                            glucose_first = lab["charttime"].timestamp()
                        if lab["charttime"].timestamp() > glucose_last:
                            glucose_last = lab["charttime"].timestamp()

                    if item.label == "Creatinine":
                        serum_time_series.append(lab.valuenum)
                        if serum_first is None:
                            serum_first = lab["charttime"].timestamp()
                            serum_last = lab["charttime"].timestamp()
                        if lab["charttime"].timestamp() < serum_first:
                            serum_first = lab["charttime"].timestamp()
                        if lab["charttime"].timestamp() > serum_last:
                            serum_last = lab["charttime"].timestamp()

                    if "A1c" in item.label:
                        hemo_time_series.append(lab.valuenum)
                        if hemo_first is None:
                            hemo_first = lab["charttime"].timestamp()
                            hemo_last = lab["charttime"].timestamp()
                        if lab["charttime"].timestamp() < hemo_first:
                            hemo_first = lab["charttime"].timestamp()
                        if lab["charttime"].timestamp() > hemo_last:
                            hemo_last = lab["charttime"].timestamp()

                if len(glucose_time_series) > 0:
                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Glucose"
                    event["desc"] = "mean"
                    event["value"] = str(discretize(np.mean(glucose_time_series), ranges_glucose))

                    if glucose_first < admittime:
                        event["timestamp"] = admittime
                    else:
                        event["timestamp"] = glucose_first
                    trace["events"].append(event)

                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Glucose"
                    event["desc"] = "std"
                    event["value"] = str(discretize(np.std(glucose_time_series), ranges_glucose_std))
                    event["timestamp"] = glucose_last
                    trace["events"].append(event)

                if len(serum_time_series) > 0:
                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Creatinine"
                    event["desc"] = "mean"
                    event["value"] = str(discretize(np.mean(serum_time_series), ranges_creatinine))

                    if serum_first < admittime:
                        event["timestamp"] = admittime
                    else:
                        event["timestamp"] = serum_first
                    trace["events"].append(event)

                    event = {}
                    event["type"] = "lab"
                    event["label"] = "Creatinine"
                    event["desc"] = "std"
                    event["value"] = str(discretize(np.std(serum_time_series), ranges_creatinine_std))
                    event["timestamp"] = serum_last
                    trace["events"].append(event)

                if len(hemo_time_series) > 0:
                    event = {}
                    event["type"] = "lab"
                    event["label"] = "A1c"
                    event["desc"] = "mean"
                    event["value"] = str(discretize(np.mean(hemo_time_series), ranges_hemo))
                    if hemo_first < admittime:
                        event["timestamp"] = admittime
                    else:
                        event["timestamp"] = hemo_first

                    trace["events"].append(event)

                    event = {}
                    event["type"] = "lab"
                    event["label"] = "A1c"
                    event["desc"] = "std"
                    event["value"] = str(discretize(np.std(hemo_time_series), ranges_hemo_std))
                    event["timestamp"] = hemo_last
                    trace["events"].append(event)

                """ Process CPT Events """
                cptevents = self.getCptEvents(admission_id)
                for _, cpt in cptevents.iterrows():
                    event = {}
                    event["type"] = "cpt"
                    event["admission_id"] = admission_id
                    if str(cpt["chartdate"]) == "NaT" or str(cpt["chartdate"]) == "None":
                        event["timestamp"] = dischtime
                        dischtime += 0.1
                    else:
                        event["timestamp"] = cpt["chartdate"].timestamp()
                    event["code"] = cpt["cpt_cd"]
                    event["cpt_number"] = cpt["cpt_number"]
                    event["cpt_suffix"] = cpt["cpt_suffix"]

                    trace["events"].append(event)

                """ Process ICD-9 Procedures """
                icdprocs = self.getIcdProcedures(admission_id)
                for _, icdproc in icdprocs.iterrows():
                    event = {}
                    event["type"] = "icd-10-pcs"
                    event["admission_id"] = admission_id
                    event["timestamp"] = dischtime
                    dischtime += 0.1
                    event["icd"] = icdproc["icd9_code"]
                    trace["events"].append(event)

                """ Process ICD-9 Diagnoses """
                diabetic_index = -1
                icddiags = self.getIcdDiagnoses(admission_id)
                cnt = 0
                for _, icddiag in icddiags.iterrows():
                    event = {}
                    event["type"] = "icd-10"
                    event["timestamp"] = dischtime
                    dischtime += 0.1
                    event["icd"] = icddiag["icd9_code"]
                    event["admission_id"] = admission_id
                    trace["events"].append(event)
                    if str(icddiag["icd9_code"]).startswith("250") and diabetic_index < 0:
                        diabetic_index = cnt
                    cnt += 1

                """ Process Comorbidities """
                dischtime += 1000.0
                if str(admission_id) in self.comorbidities.keys():
                    comorbidities = self.comorbidities[str(admission_id)]
                    for comorb in comorbidities:
                        event = {}
                        event["type"] = "elix_comorb"
                        event["timestamp"] = dischtime
                        dischtime += 0.1
                        event["name"] = comorb
                        event["admission_id"] = admission_id
                        trace["events"].append(event)
                else:
                    print("No comorb!")

                meta["diabetic_icd_index"].append(diabetic_index)
                meta["icd_indices"].append(cnt)


            trace['events'] = sorted(trace['events'], key=lambda event: event['timestamp'])

            """ SET PREDICT AT LAST ADMISSION TYPE ONE """
            if len(trace["events"]) > 0:
                cc = -1
                for i in range(len(trace["events"])):
                    if trace["events"][cc]["type"] == "admission":
                        trace["events"][cc]["PREDICT"] = True
                        print(trace["events"][cc])
                        break
                    else:
                        cc -= 1

            traces.append(trace)
            metadata[patient["subject_id"]] = meta

            trace_cnt += 1
            print("Trace Count:", trace_cnt)
        return traces, metadata

    def getDiabetesPatients(self):
        diabs = pd.read_csv("data/patients_train.csv")
        diabs2 = pd.read_csv("data/patients_test.csv")
        diabs3 = pd.read_csv("data/patients_val.csv")
        diabs = [str(diab) for diab in diabs.subject_id.unique()]
        for d in diabs2.subject_id.unique():
            diabs.append(d)
        for d in diabs3.subject_id.unique():
            diabs.append(d)

        print("Number of Patients:", len(diabs))
        query = "SELECT p.subject_id, p.gender, p.dob, p.dod FROM patients p WHERE p.subject_id IN {}".format(tuple(diabs))
        final_patients = self.db.query(query)

        return final_patients

    def getAdmissionTimes(self, admissions):
        admit = list()
        discharge = list()
        for _, admission in admissions.iterrows():
            dischtime = admission["dischtime"].timestamp()
            admittime = admission["admittime"].timestamp()
            admit.append(admittime)
            discharge.append(dischtime)
        return admit, discharge

    def getDeltasBetweenDischarges(self,admissions):
        """ Deltas are in days
        :param admissions:
        :return:
        """
        deltas = list()
        prev = None
        for _, admission in admissions.iterrows():
            dischtime = admission["dischtime"].timestamp()
            if prev is not None:
                delta = dischtime - prev
                deltas.append(delta/ (60*60*24))
            prev = dischtime
        return deltas

    def getDeltasBetweenAdmissions(self,admissions):
        """ Deltas are in days
        :param admissions:
        :return:
        """
        deltas = list()
        prev = None
        for _, admission in admissions.iterrows():
            admittime = admission["admittime"].timestamp()
            if prev is not None:
                delta = admittime - prev
                deltas.append(delta/ (60*60*24))
            prev = admittime
        return deltas

    def getDeltasBetweenDischargeAdmissions(self,admissions):
        """
        In days!
        :param admissions:
        :return:
        """
        deltas = list()
        prev = None
        for _, admission in admissions.iterrows():
            dischtime = admission["dischtime"].timestamp()
            admittime = admission["admittime"].timestamp()
            if prev is not None:
                delta = admittime - prev
                deltas.append(delta/ (60*60*24))
            prev = dischtime
        return deltas

    def inHospitalDeaths(self,admissions):
        """
        True if patient died during hospital stay
        :param admissions:
        :return:
        """
        isdead = list()
        for _, admission in admissions.iterrows():
            if str(admission["deathtime"]) != "NaT" and str(admission["deathtime"]) != "None":
                if admission["dischtime"].timestamp() == admission["deathtime"].timestamp():
                    isdead.append(True)
                else:
                    isdead.append(False)
            else:
                isdead.append(False)
        return isdead

    def getAdmissionsList(self,admissions):
        lst = list()
        for _, admission in admissions.iterrows():
            lst.append(admission["hadm_id"])
        return lst


    """ DATABASE FUNCTIONS """

    def getAdmissions(self, patient):
        query = "SELECT a.subject_id, a.hadm_id, a.dischtime, a.admittime, a.insurance, a.admission_type, a.deathtime, a.ethnicity FROM admissions a WHERE a.subject_id=" + str(
            patient) + " ORDER BY a.dischtime ASC;"
        admissions = self.db.query(query)
        return admissions

    def getLabEvents(self, admission_id):
        query = "SELECT c.subject_id, c.hadm_id, c.itemid, c.charttime, c.value, c.valuenum, c.valueuom, c.flag FROM labevents c WHERE c.hadm_id=" + str(
            admission_id) + " AND c.itemid IN (50852, 50854, 50931, 50912) ORDER BY c.charttime ASC;"
        labevents = self.db.query(query)
        return labevents

    def getLabEventsUnassigned(self, subject_id):
        query = "SELECT c.subject_id, c.hadm_id, c.itemid, c.charttime, c.value, c.valuenum, c.valueuom, c.flag FROM labevents c WHERE c.subject_id=" + str(
            subject_id) + " AND c.hadm_id IS NULL AND c.itemid IN (50852, 50854, 50931, 50912) ORDER BY c.charttime ASC;"
        labevents = self.db.query(query)
        return labevents

    def getDLabitems(self):
        query = "SELECT * FROM d_labitems WHERE itemid IN (50852, 50854, 50931, 50912);"
        return self.db.query(query)


    def getCptEvents(self, admission_id):
        query = "SELECT c.subject_id, c.hadm_id, c.costcenter, c.chartdate, c.cpt_cd, c.cpt_number, c.cpt_suffix, c.ticket_id_seq FROM cptevents c WHERE c.hadm_id=" + str(
            admission_id) + " ORDER BY c.chartdate, c.ticket_id_seq ASC;"
        cptevents = self.db.query(query)
        return cptevents

    def getDrgEvents(self, admission_id):
        query = "SELECT c.subject_id, c.hadm_id, c.drg_code, c.description, c.drg_severity, c.drg_mortality FROM drgcodes c WHERE c.hadm_id=" + str(
            admission_id) + " AND c.drg_type='HCFA';"
        drgevents = self.db.query(query)
        return drgevents

    def getIcdProcedures(self, admission_id):
        query = "SELECT d.subject_id, d.hadm_id, d.seq_num, d.icd9_code FROM procedures_icd d WHERE d.hadm_id=" + str(
            admission_id) + " ORDER BY d.seq_num ASC;"
        icdproc = self.db.query(query)
        return icdproc

    def getIcdDiagnoses(self, admission_id):
        query = "SELECT d.subject_id, d.hadm_id, d.seq_num, d.icd9_code FROM diagnoses_icd d WHERE d.hadm_id=" + str(
            admission_id) + " ORDER BY d.seq_num ASC;"
        icddiag = self.db.query(query)
        return icddiag

    def getMedications(self, admission_id):
        query = "SELECT d.drug_name_generic, d.startdate, d.enddate FROM prescriptions d WHERE d.hadm_id=" + str(
            admission_id) + " AND (d.drug_name_generic LIKE '%insulin%' OR d.drug_name_generic LIKE '%Insulin%') ORDER BY d.startdate ASC;"
        return self.db.query(query)