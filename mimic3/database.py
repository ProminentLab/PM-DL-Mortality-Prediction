import pandas as pd
import psycopg2
from config.conf import mimic_conf

class MIMIC:
    def __init__(self):
        self.con = psycopg2.connect(dbname=mimic_conf["dbname"], user=mimic_conf["user"], host=mimic_conf["host"], password=mimic_conf["password"])
        cur = self.con.cursor()
        cur.execute('SET search_path to {}'.format(mimic_conf["schema"]))
        print("Connected to MIMIC-iii")

    def query(self, sql_query):
        return pd.read_sql_query(sql_query, self.con)