import pandas as pd

class CPT:
    def __init__(self):
        self.cpt1_sections = pd.read_csv("data/util/cpt1_sections.csv")
        self.cpt2_sections = pd.read_csv("data/util/cpt2_sections.csv")
        self.cpt1_details = pd.read_csv("data/util/cpt1_services.csv")

    def details(self, cpt):
        details = dict()
        details["code"] = cpt

        if cpt.isnumeric():
            details["type"] = "CPT-I"

            details["section"] = None
            for _, row in self.cpt1_sections.iterrows():
                if int(row["start"]) <= int(cpt) and int(row["end"]) >= int(cpt):
                    details["section"] = row["desc"]

            details["subfield"] = None
            for _, row in self.cpt1_details.iterrows():
                if int(row["start"]) <= int(cpt) and int(row["end"]) >= int(cpt):
                    details["subfield"] = row["desc"]

        elif cpt[4] == "F":
            details["type"] = "CPT-II"
            details["section"] = None

            for _, row in self.cpt2_sections.iterrows():
                if int(row["start"]) <= int(cpt) and int(row["end"]) >= int(cpt):
                    details["section"] = row["desc"]

            details["subfield"] = None

        else:
            details["type"] = "CPT-III"
            details["section"] = None
            details["subfield"] = None

        return details

