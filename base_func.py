# collection of common tools

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
#from xgboost import XGBRegressor
from vecstack import stacking

def fill_nans(data, filler=-999):
    """Takes df and replace nan values with filler value (-999 by default)"""
    for col in data.columns:
        data[col].fillna(filler, inplace=True)


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

def base_reg_stack(x, y, x_test):
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
    test_size = 0.2, random_state = 0)

    # Caution! All models and parameter values are just 
    # demonstrational and shouldn't be considered as recommended.
    # Initialize 1st level models.
    models = [
        ExtraTreesRegressor(random_state = 0, n_jobs = -1, 
            n_estimators = 100, max_depth = 3),
            
        RandomForestRegressor(random_state = 0, n_jobs = -1, 
            n_estimators = 100, max_depth = 3),
            
        GradientBoostingRegressor(learning_rate = 0.1, 
            n_estimators = 100, max_depth = 3)]
    
    # Compute stacking features
    S_train, S_test = stacking(models, X_train, y_train, x_test, 
        regression = True, metric = r2_score, n_folds = 4, 
        shuffle = True, random_state = 0, verbose = 2)

    # Initialize 2nd level model
    model = GradientBoostingRegressor(learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)
    print("S_train shape:", S_train.shape)
    # Fit 2nd level model
    model = model.fit(S_train, y_train)

    # Predict
    y_pred = model.predict(S_test)

    return y_pred


def base_clf_stack(x, y, x_test):
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
    test_size = 0.2, random_state = 0)

    # Caution! All models and parameter values are just 
    # demonstrational and shouldn't be considered as recommended.
    # Initialize 1st level models.
    models = [
        ExtraTreesClassifier(random_state = 0, n_jobs = -1, 
            n_estimators = 100, max_depth = 3),
            
        RandomForestClassifier(random_state = 0, n_jobs = -1, 
            n_estimators = 100, max_depth = 3),
            
        GradientBoostingClassifier(learning_rate = 0.1, 
            n_estimators = 100, max_depth = 3)]
    
    # Compute stacking features
    S_train, S_test = stacking(models, X_train, y_train, x_test, 
        regression = True, metric = r2_score, n_folds = 4, 
        shuffle = True, random_state = 0, verbose = 2)

    # Initialize 2nd level model
    model = GradientBoostingClassifier(learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)
    print("S_train shape:", S_train.shape)
    # Fit 2nd level model
    model = model.fit(S_train, y_train)

    # Predict
    y_pred = model.predict(S_test)

    return y_pred

def adv_reg_stack(x, y, x_test, reg_models, metric=mean_absolute_error):
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
    test_size = 0.2, random_state = 0)
    
    # Compute stacking features
    S_train, S_test = stacking(reg_models, X_train, y_train, x_test, 
        regression = True, metric=metric, n_folds = 4, 
        shuffle = True, random_state = 0, verbose = 2)

    # Initialize 2nd level model
    model = GradientBoostingRegressor(learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)
    print("S_train shape:", S_train.shape)
    # Fit 2nd level model
    model = model.fit(S_train, y_train)
    # Predict
    y_pred = model.predict(S_test)
    return y_pred

def adv_clf_stack(x, y, x_test, clf_models, metric=mean_absolute_error):
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
    test_size = 0.2, random_state = 0)
    
    # Compute stacking features
    S_train, S_test = stacking(clf_models, X_train, y_train, x_test, 
        regression = True, metric=metric, n_folds = 4, 
        shuffle = True, random_state = 0, verbose = 2)

    # Initialize 2nd level model
    model = GradientBoostingClassifier(learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)
    print("S_train shape:", S_train.shape)
    # Fit 2nd level model
    model = model.fit(S_train, y_train)
    # Predict
    y_pred = model.predict(S_test)
    return y_pred