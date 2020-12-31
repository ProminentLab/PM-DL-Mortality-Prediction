import json
import numpy as np
import pandas as pd

from pydream.predictive.nap.SPLIT import SPLIT
from sklearn.metrics import confusion_matrix, roc_auc_score

if __name__== "__main__":
    model_path = "data/models/final_models"
    model_name = "log_train_scoreonly"

    split = SPLIT(net=None,
                  tss_train_file="data/output/sm_log_train_tss_train.json",
                  tss_test_file="data/output/sm_log_train_tss_test.json",
                  tss_val_file="data/output/sm_log_train_tss_val.json"
                  )
    y_pred, y_prob = split.predict_test(path=model_path, name=model_name)

    y_true = np.argmax(split.Y_test, axis=1)

    y_probscore = []
    list_of_lists = []
    for x in y_prob:
        list_of_lists.append(list(x))
        y_probscore.append(list(x)[1])

    result_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 'y_score': list_of_lists, 'y_probscore':y_probscore})
    result_df.to_csv("data/results/" + model_name + "_results.csv", index=None)


    print()
    print("*** RESULTS ***")
    print()
    conf = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf.ravel()

    print(conf)
    print("Test Accuracy:", (tp+tn) / (tp + tn + fp + fn))

    print()
    auc_micro = roc_auc_score(y_true, y_pred, average='micro', multi_class='raise')
    print("Test AUC (on binary):", auc_micro)