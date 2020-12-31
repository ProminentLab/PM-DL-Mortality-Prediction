import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import numpy as np

#https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
def loadData(df):
    X = df[["ethnicity", "admission_type", "gender", "age", "age_bracket", "insurance", "admission_location", "mean_blood_glucose", "mean_hb1ac", "elixhauser", "cci", "dcsi"]]
    y = df[["label"]]

    X["ethnicity"] = X["ethnicity"].astype('category')
    X["ethnicity"] = X["ethnicity"].cat.codes

    X["admission_type"] = X["admission_type"].astype('category')
    X["admission_type"] = X["admission_type"].cat.codes

    X["gender"] = X["gender"].astype('category')
    X["gender"] = X["gender"].cat.codes

    X["insurance"] = X["insurance"].astype('category')
    X["insurance"] = X["insurance"].cat.codes

    X["admission_location"] = X["admission_location"].astype('category')
    X["admission_location"] = X["admission_location"].cat.codes

    #is_NaN = df.isnull()
    #row_has_NaN = is_NaN.any(axis=1)
    return X.to_numpy(), y.to_numpy().ravel()

def loadSignificant(df, onehot=False):
    X = df[["admission_type", "mean_blood_glucose", "mean_hb1ac", "elixhauser", "cci", "dcsi", "age"]]
    y = df[["label"]]

    if onehot:
        one_hot = pd.get_dummies(X["admission_type"])
        X = X.drop("admission_type", axis=1)
        X = X.join(one_hot)
    else:
        X["admission_type"] = X["admission_type"].astype('category')
        X["admission_type"] = X["admission_type"].cat.codes

    return X.to_numpy(), y.to_numpy().ravel()


if __name__== "__main__":
    train = pd.read_csv("../data/util/ml_data_train.csv")
    X_train, y_train = loadSignificant(train, onehot=False)

    test = pd.read_csv("../data/util/ml_data_test.csv")
    X_test, y_test = loadSignificant(test, onehot=False)


    print("***** LOGISTIC REGRESSION *******")
    #clf_l1_LR = LogisticRegression(C=1, penalty='l1', tol=0.000001, solver='saga', class_weight="balanced", max_iter=10000000)
    #clf_l1_LR.fit(X_train, y_train)
    
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    param_grid = [
        {'classifier': [LogisticRegression()],
         'classifier__penalty': ['l1', 'l2'],
         'classifier__class_weight' : ["balanced"],
         'classifier__C': [0.1, 1.0, 1.5, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0],
         'classifier__solver': ['liblinear', 'saga']}
    ]
    clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    clf_l1_LR = clf.fit(X_train, y_train)
    

    y_prob = clf_l1_LR.predict_proba(X_test)
    y_pred = clf_l1_LR.predict(X_test)

    y_score = y_prob[:,1]

    """
    y_score = []
    for i in range(len(y_train)):
        y_score.append(y_prob[i][1])
        """

    y_probs = []
    for prob in y_prob:
        y_probs.append([prob[0],prob[1]])

    result_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_test, 'y_score': y_probs, 'y_probscore': y_score})
    result_df.to_csv("../data/results/ml_lr_results.csv", index=None)

    print("*** DONE ***")




