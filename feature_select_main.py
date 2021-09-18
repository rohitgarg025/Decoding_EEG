#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Script to get the feature ranking and electrode ranking through 
        # Method A :- Random Forest Regressor
        # Method B :- F score based Ranking
        # Method C :- Random Forest Importances approach 
# Main function

from ImportUtils import *
from TopNByFSMethods import *
from TopNByClassifier import *
from args_eeg import args as my_args

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
    getEpochedFeatures(dataset, window, stride, sfreq, label)
    if(top == "e"):
        clf = RandomForestRegressor()
        topElectrodeRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False)
        topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest')
        topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='RandomForest')
        plt.legend(["Method A","Method B", "Method C"])

        if(label == 1):
            plt.savefig(pwd + "/" + dataset + "/arousal_plots/" + "CorrectedElectrodewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
            plt.show()
            plt.clf()
        
        else:
            plt.savefig(pwd + "/" + dataset + "/plots/" + "CorrectedElectrodewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
            plt.show()
            plt.clf()    
        
    elif(top == "f"):
        clf = RandomForestRegressor()
        topFeaturesRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False)
        topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest')
        topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='RandomForest')
        if(label == 1):
            plt.legend(["Method A","Method B", "Method C"])
            plt.savefig(pwd + "/" + dataset + "/arousal_plots/" + "CorrectedFeaturewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
            plt.show()
            plt.clf()
        else:
            plt.legend(["Method A","Method B", "Method C"])
            plt.savefig(pwd + "/" + dataset + "/plots/" + "CorrectedFeaturewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
            plt.show()
            plt.clf()

