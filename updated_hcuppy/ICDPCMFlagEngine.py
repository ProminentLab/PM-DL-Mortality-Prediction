import csv

class ICDPCMFlagEngine:
    """
    THIS IS BASED ON ICD-9 CODES

    https://www.hcup-us.ahrq.gov/toolssoftware/surgflags/surgeryflags.jsp
    """

    def __init__(self):
        fn = "data/util/surgery_flags_i9_2015.csv"

        self.broad = set()
        self.narrow = set()

        with open(fn, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if row[1] == '1':
                    self.broad.add(str(row[0]))
                if row[1] == '2':
                    self.narrow.add(str(row[0]))

    def get_sflag(self, icd):
        out = "none"
        if icd in self.narrow:
            out = "narrow"
        elif icd in self.broad:
            out = "broad"
        return out

if __name__ == "__main__":
    eng = ICDPCMFlagEngine()
    print(eng.get_sflag("8582"))
