#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:50:05 2018

@author: Paul
"""

## create a single random forest  ---------------------------------------------
## and determine variable importance  -----------------------------------------

##  ===========================================================================
##  -------------------------   parameters   ----------------------------------
##  ===========================================================================
_home_directory = '/Users/Paul/Documents/kaggle/santander/'
_raw_data = '/Users/Paul/Documents/kaggle/santander/data/raw/'
_processed_data = '/Users/Paul/Documents/kaggle/santander/data/processed/'


## packages  ------------------------------------------------------------------
import os
import pandas as pd
import matplotlib
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor

## set home directory  --------------------------------------------------------
os.chdir(_home_directory)

##  ===========================================================================
##  ----------------------   load data files   --------------------------------
##  ===========================================================================

## application_{train|test}  --------------------------------------------------
train = pd.read_csv(_raw_data + "train.csv")
test = pd.read_csv(_raw_data + "test.csv")


##  ===========================================================================
##  ----------------------   review dataset   ---------------------------------
##  ===========================================================================
## target variable  -----------------------------------------------------------
train['target'].describe()
matplotlib.pyplot.hist(train['target'])


##  ===========================================================================
##  ----------------------   build single random forest   ---------------------
##  ===========================================================================
forest = RandomForestRegressor(n_estimators = 600
                            , min_samples_leaf = 10
                            )
forest.fit(y = train['target'], X = train[train.columns[2:4993]])
train_scored = pd.concat([train['target']
                , pd.Series(forest.predict(train[train.columns[2:4993]]))]
                , axis = 1
                )
## update column names  -------------------------------------------------------
train_scored.columns = ['target', 'fitted']
matplotlib.pyplot.hist(train_scored['target'])
matplotlib.pyplot.hist(train_scored['fitted'])

## variable importance  -------------------------------------------------------




##  ===========================================================================
##  ---------------------------   scratch   -----------------------------------
##  ===========================================================================
forest = RandomForestRegressor(n_estimators = 500
                            , min_samples_leaf = 10)
forest.fit(y = train['target'], X = train[importance_05.index[0:200]])

train_scored = pd.concat([train['target']
                , pd.Series(forest.predict(train[importance_05.index[0:200]]))], axis = 1)
train_scored.columns = ['actual', 'fitted']
