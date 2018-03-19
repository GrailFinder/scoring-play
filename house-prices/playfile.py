import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def fill_nans(data):
    """Takes df and replace nan values with -999"""
    for col in data.columns:
        data[col].fillna(-999, inplace=True)


def make_prediction(x_train, y_train, x_test, model):
    """Predicts the probability of serious delinquency."""
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return y_predict

def roc_score(x, y, model):
    """Estimates the area under ROC curve of a model."""
    # We use k-fold cross-validation and average the scores.
    kfold = sklearn.model_selection.KFold(n_splits=5)
    scores = []
    for train_index, test_index in kfold.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        score = sklearn.metrics.roc_auc_score(
            y_test, make_prediction(x_train, y_train, x_test, model))
        scores.append(score)
    return np.mean(scores)


def make_cats(df):
    for col in df.columns:
        if df[col].dtype == "object":
            print(df[col].dtype)
            u_vals = df[col].unique()
            k_v = {k: v for k, v in zip(range(len(u_vals)), u_vals)}
            df = df.applymap(lambda c: k_v[c] if c in k_v else c)
    return df

def catsmaker(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col].fillna('-999'))
            
    return df


if __name__ == "__main__":

    df_train = pd.DataFrame.from_csv("train.csv")
    df_test = pd.DataFrame.from_csv("test.csv")

    print(df_train.info())
    #pd.get_dummies(df_train).to_csv("dummies.csv")
    #print(pd.to_numeric(df_train['MSZoning']))


    df_train = catsmaker(df_train)
    fill_nans(df_train)
    print(df_train.columns)
    t_name = 'SalePrice'
    X = scale(df_train.drop([t_name], axis=1))
    x_train, x_test, y_train, y_test = train_test_split(X, df_train[t_name])
    # y_test = y_test.reshape(-1, 1)
    # y_train = y_train.reshape(-1, 1)
    #clf = SVR()
    clf = RandomForestRegressor()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    clf.fit(x_train, y_train)
    print(r2_score(y_test, clf.predict(x_test)))
    X_test = catsmaker(df_test)
    fill_nans(X_test)
    X_test = scale(X_test)
    submission = pd.DataFrame({
        "Id": df_test.index,
        t_name: clf.predict(X_test)
    })

    submission.to_csv('rf.csv', index=False)
