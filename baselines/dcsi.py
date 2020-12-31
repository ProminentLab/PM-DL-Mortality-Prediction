import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def loadData(df):
    X = df[["dcsi"]]
    y = df[["label"]]
    return X.to_numpy().flatten(), y.to_numpy().ravel()

if __name__== "__main__":
    train = pd.read_csv("../data/util/ml_data_train.csv")
    X_train, y_train = loadData(train)

    fpr, tpr, thresholds = roc_curve(y_train, X_train)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Train AUC:", roc_auc_score(y_train, X_train))
    print("Optimal threshold:", optimal_threshold)

    test = pd.read_csv("../data/util/ml_data_test.csv")
    X_test, y_test = loadData(test)


    y_pred = []
    y_true = []
    y_score = []
    y_prob = []
    for i in range(len(y_test)):
        y_score.append(X_test[i])
        y_true.append(y_test[i])

        if X_test[i] > optimal_threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

        y_prob.append([1-X_test[i], X_test[i]])

    result_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 'y_score': y_prob, 'y_probscore': y_score})
    result_df.to_csv("../data/results/dcsi_results.csv", index=None)

    print("*** DONE ***")