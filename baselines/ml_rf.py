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


    pipe = Pipeline([('classifier', RandomForestClassifier())])
    param_grid = [
            {'classifier': [RandomForestClassifier()],
             'classifier__n_estimators': [100, 250, 500, 1000, 1500, 2500, 5000, 10000],
             'classifier__class_weight': ["balanced"],
             'classifier__max_features': list([2,3,4,5])}
    ]
    clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf = clf.fit(X_train, y_train)
    print(best_clf)

    y_prob = best_clf.predict_proba(X_test)
    y_pred = best_clf.predict(X_test)
    y_score = y_prob[:,1]

    y_probs = []
    for prob in y_prob:
        y_probs.append([prob[0],prob[1]])

    result_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_test, 'y_score': y_probs, 'y_probscore': y_score})
    result_df.to_csv("../data/results/ml_rf_results.csv", index=None)

    print("*** DONE ***")




