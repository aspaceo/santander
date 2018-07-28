#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:58:10 2018

@author: Paul
"""

# -*- coding: utf-8 -*-
"""
loading of data files and inital review


"""

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
##  ----------------------   functions   --------------------------------------
##  ===========================================================================

## build x random forests;
## store variable importance;
## select most important variables;
## build model from those variables;

def rf_generator(df
                 , vars_to_include
                 , no_of_models
                 , no_of_trees
                 , no_of_vars
                 , target = 'target'
                 , min_samp_leaf = 30
                 , seed = 100):
    ''' 
        rf_generator will train the specified number of 
        random forests according to the parameters specified; returning
        a dataframe containing the reported variable importance for each
        variable, for each model run 
        
    '''
    
    ## internal functions  ----------------------------------------------------
    def _create_result_frame():
        nonlocal vars_to_include
        nonlocal no_of_models
        npa= np.full(shape = (len(vars_to_include), no_of_models), fill_value = -1)
        rf = pd.DataFrame(npa, index = vars_to_include)
        return(rf)


    ## build a dataframe to store results of x models  ------------------------
    rf = _create_result_frame()
    
    ## looping over random forests  -------------------------------------------
    for i in range(no_of_models):
        ## randomly select x vars to include  ---------------------------------
        mod_vars = random.sample(list(vars_to_include), no_of_vars)
        
        forest = RandomForestRegressor(n_estimators = no_of_trees
                                    , min_samples_leaf = min_samp_leaf
                                    )
        forest.fit(y = df[target], X = df[mod_vars])
        mod_score = forest.score(y = df[target], X = df[mod_vars])
        
        ## update rf to store variable importance  ----------------------------
        rf.loc[mod_vars, i] = forest.feature_importances_
        print("scored rf: " + str(i) + " - Score: " + str(mod_score))

    ##  return  ---------------------------------------------------------------
    return(rf)


##  ===========================================================================
##  ----------------------   attempt #1   -------------------------------------
##  ===========================================================================
rsts = rf_generator(df = train
                    , vars_to_include = train.columns[2:]
                    , no_of_models = 2000
                    , no_of_trees = 250
                    , no_of_vars = 50
                    , min_samp_leaf = 10
                    )

## determine mean importance for each variable  -------------------------------
importance_np = np.zeros(shape = len(rsts.index))

for i, j in enumerate(rsts.index):
    fltr = rsts.loc[j] > -1
    average = (rsts.loc[j][fltr]).mean()
    importance_np[i] = average
    print(i)
    
importance = pd.DataFrame(importance_np, index = rsts.index
                          , columns = np.array(['avg_import'])
                          )

## sort by most important  ----------------------------------------------------
importance = importance.sort_values('avg_import', ascending = False)
## filter out very low values  ------------------------------------------------
importance_05 = importance[importance['avg_import'] > 0.05]

## take top 200 variables and build a random forest from it  -----------------
forest = RandomForestRegressor(n_estimators = 500
                            , min_samples_leaf = 10)
forest.fit(y = train['target'], X = train[importance_05.index[0:200]])

train_scored = pd.concat([train['target']
                , pd.Series(forest.predict(train[importance_05.index[0:200]]))], axis = 1)
train_scored.columns = ['actual', 'fitted']
train_scored.to_csv('/Users/Paul/Documents/kaggle/santander/data/processed/' + 's_2_training.csv'
                 , index = False)



    
## score model on test data  --------------------------------------------------
rst = forest.predict(test[importance_05.index[0:200]])

rst_frame = pd.concat([test['ID'], pd.Series(rst)], axis = 1, ignore_index = True)
rst_frame.columns = ['ID', 'target']
## write results  -------------------------------------------------------------
rst_frame.to_csv('/Users/Paul/Documents/kaggle/santander/data/processed/' + 's_2.csv'
                 , index = False)


##  ===========================================================================
##  ----------------------   scratch   ----------------------------------------
##  ===========================================================================



matplotlib.pyplot.hist(train['58e2e02e6'])
matplotlib.pyplot.scatter(x = train['58e2e02e6'], y = train['target'])

forest.score(y = train['target'], X = train[importance_05.index[0:200]])



df = train
vars_to_include = train.columns[2:]  ## everything but the first two  ---------
target = 'target'
no_of_models = 2000
no_of_trees = 100
no_of_vars = 20
type(list(vars_to_include))

## testing function  ----------------------------------------------------------
rf_generator(df = train
             , vars_to_include = train.columns[2:] 
             , target = 'target'
             , no_of_models = 10
             , no_of_trees = 10
             , no_of_vars = 20
             )