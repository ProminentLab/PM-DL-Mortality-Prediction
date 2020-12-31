import json
import pandas as pd
from mimic3.database import MIMIC
from converter.variants.RawTracesMimic3_Tools import RawTracesMimic3_Tools

if __name__ == "__main__":
    print("*** Create Raw Traces and Metadata File ***")
    trace_builder = RawTracesMimic3_Tools()
    traces, meta = trace_builder.build(db=MIMIC())

    with open('data/output/traces.json', 'w') as outfile:
        json.dump(traces, outfile)

    with open('data/output/meta.json', 'w') as outfile:
        json.dump(meta, outfile)


    print("*** Split Raw Traces in train, test, and val ***")
    train_ids = pd.read_csv("data/patients_train.csv")
    val_ids = pd.read_csv("data/patients_val.csv")
    test_ids = pd.read_csv("data/patients_test.csv")
    train_ids = train_ids.subject_id.tolist()
    val_ids = val_ids.subject_id.tolist()
    test_ids = test_ids.subject_id.tolist()

    test_traces, train_traces, val_traces = [], [], []
    for trace in traces:
        if trace["patient"] in test_ids:
            test_traces.append(trace)
        elif trace["patient"] in train_ids:
            train_traces.append(trace)
        elif trace["patient"] in val_ids:
            val_traces.append(trace)

    with open('data/output/traces_test.json', 'w') as outfile:
        json.dump(test_traces, outfile)

    with open('data/output/traces_train.json', 'w') as outfile:
        json.dump(train_traces, outfile)

    with open('data/output/traces_val.json', 'w') as outfile:
        json.dump(val_traces, outfile)


    print("*** Prepare Data For Seq2Seq Learning ***")

    with open('data/output/meta.json') as json_file:
        meta = json.load(json_file)

    stan_filter = ['3601', '8754', '9634', '370', '8669', '8165', '3927', '3615', '31760', '3712', '4523', '0131',
                   '5011', '4444', '8622', '8607', '437', '8051', '526', '3614', '3893', '35400', '9656', '4652',
                   '8417', '8104', '8102', '8849', '9672', '3199', '4693', '8415', '4639', '062', '8964', '3975',
                   '9723', '3327', '4576', '8843', '0443', '4525', '3972', '8321', '9904', '5491', '5421', '9907',
                   '3605', '3772', '3812', '5749', '3401', '0689', '8609', '3726', '3404', '022', '5137', '3794',
                   '3995', '4432', '32820', '0159', '4223', '4562', '387', '5091', '4575', '4632', '966', '4701',
                   '3409', '4573', '5553', '3322', '324', '554', '4513', '4524', '0681', '0151', '5059', '3895', '3613',
                   '8848', '3606']

    for ending in ["test", "train", "val"]:
        with open("data/output/traces_" + ending + ".json") as json_file:
            traces = json.load(json_file)

        sentences = []
        labels = []
        admissions = []
        vocabulary = set()

        for trace in traces:
            patient = trace["patient"]
            in_hospital_death = meta[str(trace["patient"])]["in_hospital_death"][-1]

            current_admission = None
            sentence = ""
            for event in trace["events"]:
                if event["type"] == "cpt" or event["type"] == "icd-10-pcs" or event["type"] == "icd-10" or event[
                    "type"] == "icd-9-pcs" or event["type"] == "icd-9":
                    if event["type"] == "cpt":
                        word = event["code"]
                    else:
                        word = event["icd"]

                    if word is None or word in stan_filter:
                        continue

                    if current_admission is None:
                        current_admission = event["admission_id"]
                        sentence = sentence + word + " "
                    elif current_admission == event["admission_id"]:
                        sentence = sentence + word + " "
                    else:
                        sentences.append(sentence + ".")
                        admissions.append(current_admission)
                        if in_hospital_death:
                            labels.append(1)
                        else:
                            labels.append(0)

                        sentence = word + " "
                        vocabulary.add(word)
                        current_admission = event["admission_id"]

            # Clear buffer
            sentences.append(sentence + ".")
            admissions.append(current_admission)
            if in_hospital_death:
                labels.append(2)
            else:
                labels.append(0)

        vocabulary = list(vocabulary)

        print("Number of sentencens:", len(admissions))
        print("Vocabulary size:", len(vocabulary))

        with open('data/output/seq_sentences_' + ending + '.txt', 'w') as filehandle:
            for listitem in sentences:
                filehandle.write('%s\n' % listitem)

        with open('data/output/seq_admissions_' + ending + '.txt', 'w') as filehandle:
            for listitem in admissions:
                filehandle.write('%s\n' % listitem)

        with open('data/output/seq_labels_' + ending + '.txt', 'w') as filehandle:
            for listitem in labels:
                filehandle.write('%s\n' % listitem)

        with open('data/output/seq_vocab_' + ending + '.txt', 'w') as filehandle:
            for listitem in vocabulary:
                filehandle.write('%s\n' % listitem)

    print("*** DONE ***")