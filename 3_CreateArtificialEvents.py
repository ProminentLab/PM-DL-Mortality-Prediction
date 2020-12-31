#https://github.com/clinicalml/embeddings

from seq2seq.models.DataLoader import DataLoader
from seq2seq.models.seq2seq import Seq2Seq
import json

def process_stanford_embeddings():
    embed_file = open("data/util/IDX_IPR_C_N_L_month_ALL_MEMBERS_fold1_s300_w20_ss5_hs_thr12.txt", "r")
    cnt = 0

    new_lines = []
    for aline in embed_file:
        values = aline.split()
        if cnt < 2:
            cnt += 1
            new_lines.append(" ".join(values))
        else:
            if values[0].startswith("IDX_"):
                values[0] = values[0].replace("IDX_", "")
                values[0] = values[0].replace(".", "")
                line = " ".join(values)
                new_lines.append(line)
            elif values[0].startswith("C_"):
                values[0] = values[0].replace("C_", "")
                line = " ".join(values)
                new_lines.append(line)
    embed_file.close()

    cnt = 0
    with open('data/util/transformed.txt', 'w') as f:
        for line in new_lines:
            l = line + "\n"
            f.write(l)
            #print(len(l.split(" ")))
            if len(l.split(" ")) == 301:
                cnt +=1

    print(cnt, "embeddings.")

if __name__ == "__main__":
    print("*** Process Stanford Embeddings ***")
    process_stanford_embeddings()

    print("*** Train Seq2Seq Model ***")
    train_file = "data/output/seq_sentences_train.txt"
    train_labels_file = "data/output/seq_labels_train.txt"
    train_admissions = "data/output/seq_admissions_train.txt"

    val_file = "data/output/seq_sentences_val.txt"
    val_labels_file = "data/output/seq_labels_val.txt"
    val_admissions = "data/output/seq_admissions_val.txt"

    test_file = "data/output/seq_sentences_test.txt"
    test_labels_file = "data/output/seq_labels_test.txt"
    test_admissions = "data/output/seq_admissions_test.txt"

    embeddings_file = "data/util/transformed.txt"

    dataloader = DataLoader(train_file=train_file, embeddings_file=embeddings_file, train_labels_file=train_labels_file,
                            test_file=test_file, test_labels_file=test_labels_file,
                            train_admissions=train_admissions, test_admissions=test_admissions,
                            val_file=val_file, val_labels_file=val_labels_file, val_admissions=val_admissions)
    seq2seq = Seq2Seq(dataLoader=dataloader, train=False)

    print("*** Create Embedded Events ***")
    seq2seq.createEvents()


    print("*** Add Artificial Events to Traces ***")
    with open('data/output/seq2seq_embeddings_test.json') as json_file:
        seq2seq = json.load(json_file)
    with open('data/output/seq2seq_embeddings_train.json') as json_file:
        seq2seq.update(json.load(json_file))
    with open('data/output/seq2seq_embeddings_val.json') as json_file:
        seq2seq.update(json.load(json_file))

    with open('data/output/meta.json') as json_file:
        meta = json.load(json_file)

    for ending in ["train", "test", "val"]:
        with open('data/output/traces_' + ending + '.json') as json_file:
            traces = json.load(json_file)

        new_traces = []
        for t, trace in enumerate(traces, 0):
            process_admissions = {}
            process_admissions_predict = {}

            patient = {}
            patient["patient"] = int(trace["patient"])
            patient["gender"] = trace["gender"]
            patient["dob"] = trace["dob"]
            patient["dod"] = trace["dod"]
            patient["ethnicity"] = trace["ethnicity"]
            patient["age"] = trace["age"]
            patient["events"] = []
            for event in traces[t]["events"]:
                if event["type"] == 'cpt' or event["type"] == 'icd-10-pcs' or event["type"] == 'icd-10':
                    if event['admission_id'] not in process_admissions.keys():
                        process_admissions[event['admission_id']] = event['timestamp']
                        process_admissions_predict[event['admission_id']] = False
                    elif process_admissions[event['admission_id']] < event['timestamp']:
                        process_admissions[event['admission_id']] = event['timestamp']

                    if "PREDICT" in event.keys():
                        process_admissions_predict[event['admission_id']] = True
                else:
                    patient["events"].append(event)

            for admission in process_admissions.keys():
                init_ts = process_admissions[admission]
                if str(admission) in seq2seq.keys():
                    for i, artificial in enumerate(seq2seq[str(admission)], 0):
                        event = {}
                        event["type"] = "artificial"
                        event["name"] = artificial
                        event["timestamp"] = init_ts + (i * 0.1)
                        event["admission_id"] = str(admission)

                        if process_admissions_predict[admission] and (i + 1) == len(seq2seq[str(admission)]):
                            event["PREDICT"] = "PREDICT"

                        patient["events"].append(event)

            patient["events"] = sorted(patient["events"], key=lambda event: event['timestamp'])
            new_traces.append(patient)

        with open('data/output/traces_proc_' + ending + '.json', 'w') as outfile:
            json.dump(new_traces, outfile)

    print("*** Done ***")