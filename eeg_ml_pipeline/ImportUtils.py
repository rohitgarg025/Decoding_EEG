#!/usr/bin/env python
# coding: utf-8

# Script to import all the required libraries.<br>
# It also defines a function to make a dictionary and load the features.
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import sys
import csv
import os
import math
import glob
from scipy import io,signal
import numpy as np
import pandas as pd

import pickle
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn import feature_selection
import argparse

from sklearn.svm import SVR 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# import cuml
# from cuml.svm import SVR
# from cuml.ensemble import RandomForestRegressor
# from cuml.svm import SVC
# from cuml.ensemble import RandomForestClassifier
# from cuml.metrics import  accuracy_score


# In[ ]:


def loadFeaturesDict(dataset):
    
# input parameters :- The name of the dataset
# return :- Feature dictionary

    featuresDict = {'shannonEntropy': None,
                'ShannonRes_delta':None,
                'ShannonRes_theta':None,
                'ShannonRes_alpha':None,
                'ShannonRes_beta':None,
                'ShannonRes_gamma':None,
                'HjorthComp':None,
                'HjorthMob':None,
                'falseNearestNeighbor':None,
                'medianFreq':None,
                'bandPwr_delta':None, 
                'bandPwr_theta':None, 
                'bandPwr_alpha':None, 
                'bandPwr_beta':None, 
                'bandPwr_gamma':None,
                'stdDev':None,
                'diffuseSlowing':None,
                'spikeNum':None,
                'deltaBurstAfterSpike':None,
                'shortSpikeNum':None,
                'numBursts':None,
                'burstLenMean':None,
                'burstLenStd':None,
                'numSuppressions':None,
                'suppLenMean':None,
                'suppLenStd':None,
                'dasm_delta': None,
                'dasm_theta': None,
                'dasm_alpha': None,
                'dasm_beta': None,
                'dasm_gamma': None,
                'rasm_delta': None,
                'rasm_theta': None,
                'rasm_alpha': None,
                'rasm_beta': None,
                'rasm_gamma': None,
                }

    featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'

    featuresDict['shannonEntropy'] = np.load(featurepath + "shannonEntropy_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_delta'] = np.load(featurepath + "ShannonRes_sub_bands_delta_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_theta'] = np.load(featurepath + "ShannonRes_sub_bands_theta_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_alpha'] = np.load(featurepath + "ShannonRes_sub_bands_alpha_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_beta'] = np.load(featurepath + "ShannonRes_sub_bands_beta_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_gamma'] = np.load(featurepath + "ShannonRes_sub_bands_gamma_1_1.npz", allow_pickle=True)['features']

    featuresDict['HjorthComp'] = np.load(featurepath + "Hjorth_complexity_1_1.npz", allow_pickle=True)['features']

    featuresDict['HjorthMob'] = np.load(featurepath + "Hjorth_mobilty_1_1.npz",allow_pickle=True)['features']

    featuresDict['falseNearestNeighbor'] = np.load(featurepath + "falseNearestNeighbor_1_1.npz",allow_pickle=True)['features']

    featuresDict['medianFreq'] = np.load(featurepath + "medianFreq_1_1.npz",allow_pickle=True)['features']

    featuresDict['bandPwr_delta']=np.load(featurepath+"bandPwr_delta_1_1.npz", allow_pickle = True)['features']

    featuresDict['bandPwr_theta']=np.load(featurepath + "bandPwr_theta_1_1.npz", allow_pickle = True)['features']
    
    featuresDict['bandPwr_alpha']=np.load(featurepath + "bandPwr_alpha_1_1.npz", allow_pickle = True)['features']
    
    featuresDict['bandPwr_beta']=np.load(featurepath + "bandPwr_beta_1_1.npz", allow_pickle = True)['features']

    featuresDict['bandPwr_gamma']=np.load(featurepath + "bandPwr_gamma_1_1.npz", allow_pickle = True)['features']

    featuresDict['stdDev'] = np.load(featurepath + "stdDev_1_1.npz",allow_pickle=True)['features']

    featuresDict['diffuseSlowing'] = np.load(featurepath + "diffuseSlowing_1_1.npz",allow_pickle=True)['features']

    featuresDict['spikeNum'] = np.load(featurepath + "spikeNum_1_1.npz",allow_pickle=True)['features']

    featuresDict['deltaBurstAfterSpike'] = np.load(featurepath + "deltaBurstAfterSpike_1_1.npz",allow_pickle=True)['features']

    featuresDict['shortSpikeNum'] = np.load(featurepath + "shortSpikeNum_1_1.npz", allow_pickle=True)['features']

    featuresDict['numBursts'] = np.load(featurepath + "numBursts_1_1.npz",allow_pickle=True)['features']

    featuresDict['burstLenMean'] = np.load(featurepath + "burstLen_u_and_sigma_mean_1_1.npz",allow_pickle=True)['features']

    featuresDict['burstLenStd'] = np.load(featurepath + "burstLen_u_and_sigma_std_1_1.npz",allow_pickle=True)['features']

    featuresDict['numSuppressions'] = np.load(featurepath + "numSuppressions_1_1.npz",allow_pickle=True)['features']

    featuresDict['suppLenMean'] = np.load(featurepath + "suppressionLen_u_and_sigma_mean_1_1.npz",allow_pickle=True)['features']

    featuresDict['suppLenStd'] = np.load(featurepath + "suppressionLen_u_and_sigma_std_1_1.npz",allow_pickle=True)['features']

    featuresDict['dasm_delta'] = np.load(featurepath + "dasm_delta_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['dasm_theta'] = np.load(featurepath + "dasm_theta_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['dasm_alpha'] = np.load(featurepath + "dasm_alpha_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['dasm_beta'] = np.load(featurepath + "dasm_beta_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['dasm_gamma'] = np.load(featurepath + "dasm_gamma_1_1.npz",allow_pickle=True)['features']

    featuresDict['rasm_delta'] = np.load(featurepath + "rasm_delta_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['rasm_theta'] = np.load(featurepath + "rasm_theta_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['rasm_alpha'] = np.load(featurepath + "rasm_alpha_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['rasm_beta'] = np.load(featurepath + "rasm_beta_1_1.npz",allow_pickle=True)['features']
    
    featuresDict['rasm_gamma'] = np.load(featurepath + "rasm_gamma_1_1.npz",allow_pickle=True)['features']

    return featuresDict

