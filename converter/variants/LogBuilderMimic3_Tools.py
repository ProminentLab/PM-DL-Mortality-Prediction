import pandas as pd
from mimic3.database import MIMIC
from icd9.icd9 import ICD9
from opyenxes.factory.XFactory import XFactory
from hcuppy.prcls import PrClsEngine
from hcuppy.cci import CCIEngine
from cpt.cpt import CPT
from updated_hcuppy.ICDPCMFlagEngine import ICDPCMFlagEngine
from updated_hcuppy.SFlagEngine import SFlagEngine

""" Charlson Comorbidity Index and Elixhauser Comorbidity Index"""
#https://github.com/mark-hoffmann/icd

""" Diabetic Severity Index """

pce = PrClsEngine()
ce = CCIEngine()

tree = ICD9('icd9/new_codes.json')
tree_alt = ICD9('icd9/procs_codes.json')

def getICDetails(icd):
    if icd == None:
        return None

    if len(icd) > 3:
        icd = icd[:3] + '.' + icd[3:]
    code_node = tree.find(icd)

    if code_node == None:
        code_node = tree_alt.find(icd)

    if code_node == None:
        code_node = tree.find(icd[:3])

    if code_node == None:
        code_node = tree_alt.find(icd[:3])

    if code_node == None:
        return None

    details = dict()
    details["icd9"] = code_node.code

    if code_node is not None:
        details["desc"] = str(code_node.description)

        node = code_node
        while node is not None:
            depth = node.depth

            if depth == 0:
                details["section"] = str(node.code)
                details["section_desc"] = str(node.description)

            if depth == 1:
                details["chapter"] = str(node.code)
                details["chapter_desc"] = str(node.description)

            elif depth == 2:
                details["disease"] =  str(node.code)
                details["disease_desc"] = str(node.description)

            elif depth == 3:
                details["clinical"] =  str(node.code)
                details["clinical_desc"] = str(node.description)

            elif depth == 4:
                details["extension"] =  str(node.code)
                details["extension_desc"] = str(node.description)
            node = node.parent
    else:
        details["desc"] = "None"
        details["chapter"] = "None"
        details["chapter_desc"] = "None"
        details["disease"] = "None"
        details["disease_desc"] = "None"
        details["clinical"] = "None"
        details["clinical_desc"] = "None"
        details["extension"] = "None"
        details["extension_desc"] = "None"

    return details

class LogBuilderMimic3_Tools:
    def __init__(self):
        self.db = MIMIC()
        self.cpt = CPT()

        self.icdpcs_flagger = ICDPCMFlagEngine()
        self.cpt_flagger = SFlagEngine()

        self.comorbidities = self.loadComorbidities()
        self.elix = pd.read_csv("data/util/patients_elix.csv")
        self.charlson = pd.read_csv("data/util/patients_charlson.csv")

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

    def build(self, traces, metadata):
        self.rawTraces = traces
        self.metadata = metadata
        return self.transformRawTracesToCodeXes()

    def getComorbidityIndices(self, patient):
        cci = self.charlson.loc[self.charlson['patient'] == patient].iloc[0]["wscore"]
        elix = self.elix.loc[self.elix['patient'] == patient].iloc[0]["wscore_ahrq"]
        return cci, elix

    def transformRawTracesToCodeXes(self):
        log = XFactory.create_log()

        trace_cnt = 0
        for trace in self.rawTraces:
            print("new Trace ", trace_cnt)
            trace_cnt += 1
            t = XFactory.create_trace()
            cci, elix = self.getComorbidityIndices(trace["patient"])

            t.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name", trace["patient"])
            t.get_attributes()["gender"] = XFactory.create_attribute_literal("gender", trace["gender"])
            t.get_attributes()["age"] = XFactory.create_attribute_literal("age", trace["age"])
            t.get_attributes()["charlson"] = XFactory.create_attribute_literal("charlson", cci)
            t.get_attributes()["elixhauser"] = XFactory.create_attribute_literal("elixhauser", elix)
            t.get_attributes()["dob"] = XFactory.create_attribute_timestamp("dob", int(trace["dob"]*1000))
            t.get_attributes()["dod"] = XFactory.create_attribute_literal("dod", str(trace["dod"]))
            t.get_attributes()["ethnicity"] = XFactory.create_attribute_literal("ethnicity", str(trace["ethnicity"]))

            label = self.metadata[str(trace["patient"])]["in_hospital_death"][-1]
            t.get_attributes()["label"] = XFactory.create_attribute_literal("label", str(label))

            for event in trace["events"]:
                e = XFactory.create_event()
                e.get_attributes()["type"] = XFactory.create_attribute_literal("type", event["type"])

                if "PREDICT" in event.keys():
                    e.get_attributes()["PREDICT"] = XFactory.create_attribute_literal("PREDICT", event["PREDICT"])

                if event["type"] == "admission":
                    activity = str(event["value"])
                    if activity == "URGENT" or activity == "EMERGENCY":
                        activity = "unplanned"
                    elif activity == "NEWBORN" or activity == "ELECTIVE":
                        activity = "planned"
                    try:
                        e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name", activity)
                        e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "admission")
                        e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id", event["admission_id"])
                        e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))
                    except:
                        print("Error")

                elif event["type"] == "lab":
                    try:
                        e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name", event["label"] + "_" + event["desc"] + "_" + str(event["value"]))
                        e.get_attributes()["value"] = XFactory.create_attribute_literal("value",event["value"])
                        e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "lab")
                        e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))
                    except:
                        print("Error")

                elif event["type"] == "cpt":
                    try:
                        details = self.cpt.details(str(event["code"]))
                        is_surgical = self.cpt_flagger.get_sflag(str(event["code"]))

                        if str(details["section"]) == "evaluation and management" or is_surgical == "none":
                            continue
                        e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name",str(details["section"]) + "_" + is_surgical)
                        e.get_attributes()["is_surgical"] = XFactory.create_attribute_literal("is_surgical", is_surgical)
                        e.get_attributes()["code"] = XFactory.create_attribute_literal("code", str(event["code"]))
                        e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id", str(event["admission_id"]))
                        e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))
                    except:
                        print("No CPT details")

                elif event["type"] == "artificial":
                    e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name", event["name"])
                    e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "artificial")
                    e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id", event["admission_id"])
                    e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))

                elif event["type"] == "icd-10":
                    continue
                    details = getICDetails(event["icd"])
                    if details == None:
                        continue
                    if "disease" in details.keys() and str(event["icd"]).startswith("250"):
                        activity = str(details["disease"])
                    elif "chapter" in details.keys():
                        activity = str(details["chapter"])
                    else:
                        activity = str(details["section"])

                    e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name", activity)
                    e.get_attributes()["code"] = XFactory.create_attribute_literal("code", str(event["icd"]))
                    e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "icd-9")
                    e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id",str(event["admission_id"]))
                    e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))

                elif event["type"] == "icd-10-pcs":
                    details = getICDetails(event["icd"])
                    is_surgical = self.icdpcs_flagger.get_sflag(str(event["icd"]))

                    if is_surgical != "none":
                        if details == None:
                            continue
                        if "chapter" in details.keys(): ## war disease zuvor
                            activity = str(details["chapter"]) ## war disease zuvor
                        else:
                            continue

                        e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name",activity + "_" + is_surgical)
                        e.get_attributes()["is_surgical"] = XFactory.create_attribute_literal("is_surgical", is_surgical)
                        e.get_attributes()["code"] = XFactory.create_attribute_literal("code", str(event["icd"]))
                        e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "icd-9-pcs")
                        e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id", str(event["admission_id"]))
                        e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))

                    else:
                        continue


                elif event["type"] == "elix_comorb":
                    e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name",str(event["name"]))
                    e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "elix_comorb")
                    e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id", str(event["admission_id"]))
                    e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))

                elif event["type"] == "ccs":
                    e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name",str(event["name"]))
                    e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "ccs")
                    e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id", str(event["admission_id"]))
                    e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))

                elif event["type"] == "drg":
                    continue
                    e.get_attributes()["concept:name"] = XFactory.create_attribute_literal("concept:name",str(event["code"]))
                    e.get_attributes()["type"] = XFactory.create_attribute_literal("type", "drg")
                    e.get_attributes()["admission_id"] = XFactory.create_attribute_literal("admission_id", str(event["admission_id"]))
                    e.get_attributes()["time:timestamp"] = XFactory.create_attribute_timestamp("time:timestamp",int(event["timestamp"] * 1000))

                else:
                    continue

                t.append(e)
            log.append(t)
        return log
