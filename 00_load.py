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
train['target'].describe()

train.iloc[:, 2].describe()

