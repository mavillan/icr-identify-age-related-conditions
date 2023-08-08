# %% [code]
# %% [code]
import numpy as np
import pandas as pd
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing


def load_data():
    train = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
    test  = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")
    greeks = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/greeks.csv")

    train.columns = [col.strip() for col in train.columns]
    test.columns = [col.strip() for col in test.columns]

    # available features
    input_cols = train.columns[1:-1]
    categ_cols = ["EJ"]

    # we extend train with dummies from greeks
    dummies = pd.get_dummies(greeks[["Alpha","Beta","Gamma","Delta"]])
    train[dummies.columns] = dummies

    # encode of categorical features
    encoder = preprocessing.LabelEncoder().fit(train["EJ"])
    train["EJ"] = encoder.transform(train["EJ"]).astype(int)
    test["EJ"] = encoder.transform(test["EJ"]).astype(int)
    
    return train,test,input_cols


def scale_data(train, test, input_cols, impute_only=False):
    train = train.copy()
    test = test.copy()
    
    steps = [
        ("imputer", impute.SimpleImputer(strategy="median")), 
    ]
    if not impute_only:
        steps.append(("scaler", preprocessing.MaxAbsScaler()))
    
    preproc_pipe = pipeline.Pipeline(steps)
    preproc_pipe.fit(train[input_cols])

    train[input_cols] = preproc_pipe.transform(train[input_cols])
    test[input_cols] = preproc_pipe.transform(test[input_cols])
    
    return train,test,preproc_pipe
