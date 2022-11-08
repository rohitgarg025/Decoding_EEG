#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Script to get the feature ranking and electrode ranking through 
        
# Method :- F score based Ranking
        
# Main function

from ImportUtils import *
from TopNByFSMethods import *
from TopNByClassifier import *
from args_eeg import args as my_args
# uncomment to extract features
# from EpochedFeatures import *
if __name__ == '__main__':

    # args object to fetch command line inputs
    args = my_args()
    print(args.__dict__)
    pwd = os.getcwd()

    dataset = args.dataset
    window = args.window
    stride = args.stride
    sfreq = args.sfreq
    model = args.model
    label = args.label 
    approach = args.approach #byclassifier or byfs
    ml_algo = args.ml_algo #classification or regression
    top = args.top #e or f or ef
    fs_method = args.fs_method

    #feature extraction
    # uncomment to extract features
    # getEpochedFeatures(dataset, window, stride, sfreq, label)
    if(top == "e"):
        clf = RandomForestRegressor()
        topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest')
        
    elif(top == "f"):
        clf = RandomForestRegressor()
        topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest')