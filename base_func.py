# collection of common tools

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, r2_score

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
    kfold = KFold(n_splits=5)
    scores = []
    for train_index, test_index in kfold.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        score = roc_auc_score(
            y_test, make_prediction(x_train, y_train, x_test, model))
        scores.append(score)
    return np.mean(scores)


def make_cats(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col].fillna('-999'))
            
    return df