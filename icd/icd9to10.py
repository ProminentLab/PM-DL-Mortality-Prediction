import csv, sys

class Icd9to10:
    def __init__(self):
        try:
            dict = open('data/util/ICD_9_10_d_v1.1.csv')
        except FileNotFoundError:
            print( "Unable to find file.")
            sys.exit(1)
        csv_dict = csv.reader(dict, delimiter='|')
        self.csv_dict = list(csv_dict)

    def icd9to10(self, code, isString=False):
        """ Insert dot to fulfill format """
        if not isString:
            if len(str(code)) > 3:
                code = str(code)[:3] + '.' + str(code)[3:]
            else:
                code = str(code)

        """ Find  ICD-10"""
        icd10s = set()
        for row in self.csv_dict:
            if row[1] == code:
                icd10s.add(row[0])

        return icd10s