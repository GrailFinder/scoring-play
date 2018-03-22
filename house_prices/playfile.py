import numpy as np
import pandas as pd
import sklearn, os
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from os import sys, path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_func import fill_nans, make_prediction, roc_score, make_cats


if __name__ == "__main__":

    df_train = pd.DataFrame.from_csv("train.csv")
    df_test = pd.DataFrame.from_csv("test.csv")

    print(df_train.info())
    #pd.get_dummies(df_train).to_csv("dummies.csv")
    #print(pd.to_numeric(df_train['MSZoning']))


    df_train = make_cats(df_train)
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
    X_test = make_cats(df_test)
    fill_nans(X_test)
    X_test = scale(X_test)
    submission = pd.DataFrame({
        "Id": df_test.index,
        t_name: clf.predict(X_test)
    })

    submission.to_csv('rf.csv', index=False)
