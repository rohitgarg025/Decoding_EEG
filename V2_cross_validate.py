"""
Files required:
    cross_validate.py
    DREAMER/data_extracted/featuresDict/
    DEAP/data_extracted/featuresDict/
    OASIS/data_extracted/featuresDict/
"""

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

#import machine learning algorithms
from sklearn.svm import SVR 
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import tree #DTR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor

def loadFeaturesDict(dataset):
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

    #lyapunov
    # featuresDict['hFD'] = np.load(featurepath + d/DEAP_hFD.npy",allow_pickle=True)['features']

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

    # featuresDict['regularity'] = np.load(featurepath + "",allow_pickle=True)['features']

    # featuresDict['volt05'] = np.load(featurepath + "",allow_pickle=True)['features']
    # featuresDict['volt10'] = np.load(featurepath + "",allow_pickle=True)['features']
    # featuresDict['volt20'] = np.load(featurepath + "",allow_pickle=True)['features']


    featuresDict['diffuseSlowing'] = np.load(featurepath + "diffuseSlowing_1_1.npz",allow_pickle=True)['features']

    featuresDict['spikeNum'] = np.load(featurepath + "spikeNum_1_1.npz",allow_pickle=True)['features']

    featuresDict['deltaBurstAfterSpike'] = np.load(featurepath + "deltaBurstAfterSpike_1_1.npz",allow_pickle=True)['features']

    featuresDict['shortSpikeNum'] = np.load(featurepath + "shortSpikeNum_1_1.npz", allow_pickle=True)['features']

    featuresDict['numBursts'] = np.load(featurepath + "numBursts_1_1.npz",allow_pickle=True)['features']

    featuresDict['burstLenMean'] = np.load(featurepath + "burstLen_u_and_sigma_mean_1_1.npz",allow_pickle=True)['features']

    featuresDict['burstLenStd'] = np.load(featurepath + "burstLen_u_and_sigma_std_1_1.npz",allow_pickle=True)['features']

    # featuresDict['burstBandPowers'] = np.load(featurepath + "",allow_pickle=True)['features']

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

np.random.seed(42)

def cross_validate(dataset, window, stride, sfreq, label, best_features_list):
    # Parameters :-
          # dataset :- Name of the Dataset
          # window :- Length of the sliding window in seconds
          # stride :- Stride of the sliding window in seconds
          # sfreq :- sampling frequency of the EEG dataset
          # best_features_list :- Featrue list after performing top electrode and feature analysis for various datasets
    pwd = os.getcwd()
    fs = sfreq

    featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'
    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']

    #load saved epoched features
    featuresDict = None
    featuresDict = loadFeaturesDict(dataset)

    # pop out not best features
    for k in list(featuresDict.keys()):
        if k not in best_features_list:
            
            featuresDict.pop(k)

    featuresList = list(featuresDict.keys())
    print(featuresList)

    #make feature matrix with select best features
    featureMatrix = np.empty((0,ans.shape[1])) #[14*32 + 1,80640]
    for key,value in featuresDict.items():
        featureMatrix = np.append(featureMatrix,value,axis=0)

    #remove NaN features
    if np.isnan(featureMatrix).any():
        featureMatrix = np.nan_to_num(featureMatrix,nan=0)

    #set datatype of feature matrix
    featureMatrix = featureMatrix.astype('float64')

    #transpose feature matrix to prepare X
    X = pd.DataFrame(featureMatrix.T)
    #replace infinity with NaN value and fill it with zero
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X = X.astype(np.float32)

    #convert ndarray to dataframe
    Y_epoch = pd.DataFrame(Y_epoch)

    print("Number of feature vectors in X = ", X.shape[1])
    print("X.shape = " ,X.shape)


    #***********************************************************
    #Leave-one-subject-out-CV
    #number of folds = numbParticipants
    numbParticipants = 0
    numbRecordings = 0

    if(dataset == 'DEAP'):
        numbParticipants = 32
        numbRecordings = 40
    elif(dataset == 'DREAMER'):
        # Dreamer dataset has 23 subjects, each subject was shown 18 videos 
        numbParticipants = 23
        numbRecordings = 18
    elif(dataset == 'OASIS'):
        numbParticipants = 15
        numbRecordings = 40
        

    #numbEpochs
    numbEpochs = X.shape[0]//(numbParticipants*numbRecordings)
    print(X.shape[0])
    print("numbParticipants = ", numbParticipants)
    print("numbRecordings = " , numbRecordings)
    print("numbEpochs = ", numbEpochs)
    print(type(X))
    print(type(Y_epoch))

    cv_rmse = []

    for i in range(numbParticipants):
        s = i*numbRecordings*numbEpochs
        e = (i+1)*numbRecordings*numbEpochs

        X_test = copy.deepcopy(X.iloc[s:e, :])
        y_test = copy.deepcopy(Y_epoch.iloc[s:e, label])

        X_train = copy.deepcopy(X.iloc[:s, :])
        X_train = np.append(X_train, X.iloc[e:, :],axis=0)

        y_train = copy.deepcopy(Y_epoch.iloc[:s, label])
        y_train = np.append(y_train, Y_epoch.iloc[e:, label],axis=0)

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        clf = RandomForestRegressor()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        
        rmse = mean_squared_error(y_test, y_predict,squared=False)
        print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        cv_rmse.append(rmse)
        

    
    
    print(cv_rmse)
    output_file = 'output.csv';
    with open(output_file,'a') as fd:
        fd.write(f'{dataset}_{label},')
        writer = csv.writer(fd)
        writer.writerow(cv_rmse)
        fd.write('\n')


if __name__ == '__main__':
    """##CV-Method B-Valence Label

    ####DREAMER
    """

    best_features_list = ['ShannonRes_gamma','ShannonRes_beta','HjorthMob','HjorthComp','bandPwr_alpha','stdDev','ShannonRes_alpha','bandPwr_beta','rasm_beta','dasm_beta','dasm_alpha']
    cross_validate(dataset = 'DREAMER', window = 1, stride = 1, sfreq=128, label = 0, best_features_list = best_features_list)

    """####DEAP"""

    best_features_list = ['rasm_gamma','rasm_beta','stdDev','dasm_beta','ShannonRes_beta','dasm_gamma','bandPwr_beta','ShannonRes_gamma']
    cross_validate(dataset = 'DEAP', window = 1, stride = 1, sfreq=128, label = 0, best_features_list = best_features_list)

    """####OASIS"""

    best_features_list = ['ShannonRes_gamma', 'HjorthMob']
    cross_validate(dataset = 'OASIS', window = 1, stride = 1, sfreq=128, label = 0, best_features_list = best_features_list)

    """##CV-Method-B-Arousal Label

    ####DREAMER
    """

    best_features_list = ['ShannonRes_gamma','HjorthComp','ShannonRes_beta','dasm_alpha','rasm_alpha','HjorthMob','bandPwr_alpha','stdDev','bandPwr_beta']
    cross_validate(dataset = 'DREAMER', window = 1, stride = 1, sfreq=128, label = 1, best_features_list = best_features_list)

    """####DEAP"""

    best_features_list = ['rasm_beta','rasm_gamma','bandPwr_gamma','dasm_beta','dasm_gamma','ShannonRes_beta','stdDev','bandPwr_beta','ShannonRes_gamma']
    cross_validate(dataset = 'DEAP', window = 1, stride = 1, sfreq=128, label = 1, best_features_list = best_features_list)

    """####OASIS"""

    best_features_list = ['HjorthMob']
    cross_validate(dataset = 'OASIS', window = 1, stride = 1, sfreq=128, label = 1, best_features_list = best_features_list)

