#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 00:12:47 2018

@author: Paul
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
##  ----------------------   feature engineering   ----------------------------
##  ===========================================================================

## entries greater than zero  -------------------------------------------------
train.apply(np.sum, axis=1)

train.iloc[:, 2:]

def row_sum(df):
    '''
    desc: calculate the total value of all entries for each row
    
    input: dataframe
    
    output: pandas series
    '''
    
    return(df.apply(np.sum, axis = 1))


def greater_than(df, gt = 0):
    '''
    desc: calculate the number of entries in excess of the parameter 'gt'
    
    input: dataframe
    
    output: pandas series
    '''
    return((df > gt).apply(np.sum, axis = 1))
    
    
df = train.iloc[:, 2:]
gt = 0


## function trial

row_sum(df = train.iloc[:, 2:])
greater_than(df = train.iloc[:, 2:], gt = 10000000)

##  ===========================================================================
##  ---------------------------   scratch   -----------------------------------
##  ===========================================================================

## ideas for derived vars  ----------------------------------------------------

## entries greater than zero
## row sum
## max value
## mean non zero value
## is the var value greater than zero?
## number of entries in excess of a centain threshold
## how unusual is it to have an entry in that column
## how unusual is the aggregate entries?

## can YOU explain why someone is high and someone is low?

