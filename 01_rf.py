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

## create dataframe of variable importance  -----------------------------------
imp = pd.DataFrame(forest.feature_importances_, index = train.columns[2:4993]
            , columns = np.array(['var_importance'])
            )
## sort the importance  -------------------------------------------------------
imp = imp.sort_values(by = ['var_importance'], ascending = False)
## create a subset of train limited to the 100 most important columns  --------
train_imp_var = train[['target'] + imp.iloc[0:100].index.tolist()]


## write out train_imp_var ----------------------------------------------------
train_imp_var.to_csv('/Users/Paul/Documents/kaggle/santander/data/processed/' + 'train_imp_var.csv'
                 , index = False
                 )


##  ===========================================================================
##  -------------  score single random forest on test set   -------------------
##  ===========================================================================


## generate a submission file from a single random forest  --------------------
## score model on test data  --------------------------------------------------
rst = forest.predict(test[train.columns[2:4993]])

rst_frame = pd.concat([test['ID'], pd.Series(rst)], axis = 1, ignore_index = True)
rst_frame.columns = ['ID', 'target']
## write results  -------------------------------------------------------------
rst_frame.to_csv('/Users/Paul/Documents/kaggle/santander/data/processed/' + 's_3.csv'
                 , index = False)






##  ===========================================================================
##  ---------------------------   scratch   -----------------------------------
##  ===========================================================================

## ideas for derived vars  ----------------------------------------------------

## entries greater than zero
## max value
## mean non zero value
## is the var value greater than zero?
## number of entries in excess of a centain threshold
## how unusual is it to have an entry in that column
## how unusual is the aggregate entries?

## can YOU explain why someone is high and someone is low?

##  ===========================================================================

forest = RandomForestRegressor(n_estimators = 500
                            , min_samples_leaf = 10)
forest.fit(y = train['target'], X = train[importance_05.index[0:200]])

train_scored = pd.concat([train['target']
                , pd.Series(forest.predict(train[importance_05.index[0:200]]))], axis = 1)
train_scored.columns = ['actual', 'fitted']

## what is the distribution of the target variable??  -------------------------
train['target'].describe()
matplotlib.pyplot.hist(train['target'])

rst_frame['target'].describe()