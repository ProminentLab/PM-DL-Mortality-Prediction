import csv

class SFlagEngine:
    """
    https://www.hcup-us.ahrq.gov/toolssoftware/surgeryflags_svcproc/surgeryflagssvc_proc.jsp
    """
    def __init__(self):
        fn = "data/util/surgery_flags_cpt_2017.csv"

        self.broad = set()
        self.narrow = set()

        with open(fn, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                start = row[0].split("-")[0]
                end = row[0].split("-")[1]
                for ind in range(int(start), int(end)):
                    if row[1] == '2':
                        self.narrow.add(str(ind))
                    elif row[1] == '1':
                        self.broad.add(str(ind))

    def get_sflag(self, cpt):
        out = "none"
        if cpt in self.narrow:
            out = "narrow"
        elif cpt in self.broad:
            out = "broad"
        return out



if __name__ == "__main__":
    eng = SFlagEngine()
    print(eng.get_sflag("10080"))



