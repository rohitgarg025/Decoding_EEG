#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ImportUtils import *

from sklearn.ensemble import RandomForestRegressor as sklearnrfi
from sklearn.feature_selection import VarianceThreshold


# In[ ]:


def topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest'):
    '''
    Ranks of features according to rmse computed by F score based regression
    Plots electrode v/s rmse graph
    
    '''
    # parameters :-
                # dataset - name of the dataset
                # window - length of the sliding window in seconds
                # stride - length of the stride of the sliding window in seconds
                # sfreq - sampling frequency of the EEG data
                # clf - name of the classifier to be used
                # label - valence/arousal/dominance/liking label (shape depends upon the dataset)
                # scale - sclaing of the EEG data if required
                # mutual_info - Mutual ranking between features based on information theory
                # method - 'RandomForest' 'RFE' 'SelectKBest'
                
    # returns :-
                # void
    pwd = os.getcwd()
    fs = sfreq
    electrodeList = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'

    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']

    print("Number of segments are: {}".format(ans.shape[1]))


    featuresDict = None
    featuresDict = loadFeaturesDict(dataset)
    asm_features = ['dasm_delta', 'dasm_theta', 'dasm_alpha', 'dasm_beta', 'dasm_gamma', 'rasm_delta', 'rasm_theta', 'rasm_alpha', 'rasm_beta', 'rasm_gamma']
    for asm in asm_features:
        featuresDict.pop(asm)
    
    common = []
    with open('intersection.pkl', 'rb') as f:
        common = pickle.load(f)

    for k in list(featuresDict.keys()):
        if k not in common:
            # pop out common feature
            featuresDict.pop(k)

    featuresList = list(featuresDict.keys())
    print(featuresList)

    featureMatrix = np.empty((0,ans.shape[1])) #[14*32 + 1,80640]
    for key,value in featuresDict.items():
        featureMatrix = np.append(featureMatrix,value,axis=0)


    if np.isnan(featureMatrix).any():
        featureMatrix = np.nan_to_num(featureMatrix,nan=0)

    featureMatrix = featureMatrix.astype('float64')


    feature_channel_index = []
    for feature in featuresList:
        for i in range(featuresDict[feature].shape[0]):
            if(i>=10):
                feature_channel_index.append(feature + str(i))
            else:
                feature_channel_index.append(feature + '0' + str(i))
    
    
    print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

    X = pd.DataFrame(featureMatrix.T)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X.columns = feature_channel_index
    
    #################################################################
    y = copy.deepcopy(Y_epoch[:,label]) #valence
    print("y.shape: ", y.shape)
    

    dfscores = None

    if(method == 'RandomForest'):
        '''Random Forest Feature Importances'''
        # estimator = RandomForestRegressor()
        estimator = sklearnrfi()
        fit = estimator.fit(X,y)
        dfscores = pd.DataFrame(fit.feature_importances_)
    elif(method == 'RFE'):
        ''' RFE'''
        selector = RFE(clf, n_features_to_select=X.shape[1], step=1)
        selector = selector.fit(X, y)
        dfscores = pd.DataFrame(selector.ranking_)

    elif(method == 'SelectKBest'):
        """SelecKBest"""
        #apply SelectKBest class to extract top 10 best features
        func = None
        if mutual_info == False:
            func = f_classif
        else:
            func = mutual_info_classif

        bestfeatures = SelectKBest(score_func=func, k=X.shape[1])
        fit = bestfeatures.fit(X,y)

        dfscores = pd.DataFrame(fit.scores_)

    
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    features_result = featureScores.nlargest(X.shape[1],'Score')
    print(features_result)
    features_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "CommonElectrodeFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")


    ###################################################################
    topcolumns = features_result['Specs'].values
    topfeatures = []
    topelectrodes = []

    for col in topcolumns:
        feature = col[:-2]
        electrode = int(col[-2:])
        if(feature not in topfeatures):
            topfeatures.append(feature)
        
        if(electrode not in topelectrodes):
            topelectrodes.append(electrode)
    
    ##################################################################################
    
    N =  len(topelectrodes)
    topRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]

    
    for n in range(1,N+1):
        
        electrode_index = topelectrodes[:n]
        print(topelectrodes)
        print(electrode_index)
        # X-Values
        featureMatrix = np.empty((len(featuresList)*len(electrode_index),ans.shape[1]))

        i = 0
        for index in electrode_index:
            for key,value in featuresDict.items():
                featureMatrix[i,:] = value[index,:]
                i = i+1
            
            # i = i+1
        
        featureMatrix = featureMatrix.astype(np.float32)
        print(featureMatrix.T.shape)
        
        # Removing NaN Values
        if np.isnan(featureMatrix).any():
            featureMatrix = np.nan_to_num(featureMatrix,nan=0)

    
        feature_channel_index = []
        for index in electrode_index:
            for feature in featuresList:
                feature_channel_index.append(feature + str(index))

        print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

        X = pd.DataFrame(featureMatrix.T)
        X.columns = feature_channel_index
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        

        print("Features Ready for undergoing selection tests done ...\n")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Apply classfier
        # clf = xgb.XGBClassifier(verbose = 5)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict,squared=False)
        print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        topRmseList.append(rmse)





    # features_result = features_result.reset_index()
    topNElectrode_df = pd.DataFrame(topNList)
    topNRmse_df = pd.DataFrame(topRmseList)
    #concat two dataframes for better visualization 
    topNElectrodeRanking = pd.concat([topNElectrode_df, topNRmse_df],axis=1)
    topNElectrodeRanking.columns = ['Electrode','RMSE']  #naming the dataframe columns
    print(topNElectrodeRanking)
    topNElectrodeRanking.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topCommonElectrodeFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")
    # return features_result
    
    
    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('Top N Electrodes')
    plt.ylabel('RMSE')
    plt.plot(topNElectrodeRanking.loc[:,"Electrode"], topNElectrodeRanking.loc[:,"RMSE"])
    plt.tight_layout()


# In[ ]:


def topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest'):

    # parameters :-
                # dataset - name of the dataset
                # window - length of the sliding window in seconds
                # stride - length of the stride of the sliding window in seconds
                # sfreq - sampling frequency of the EEG data
                # clf - name of the classifier to be used
                # label - valence/arousal/dominance/liking label (shape depends upon the dataset)
                # scale - sclaing of the EEG data if required
                # mutual_info - Mutual ranking between features based on information theory
                # method - 'RandomForest' 'RFE' 'SelectKBest'
                
    # returns :-
                # void
    
    pwd = os.getcwd()
    fs = sfreq
    featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'

    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']
    print("Number of segments are: {}".format(ans.shape[1]))


    featuresDict = None
    featuresDict = loadFeaturesDict(dataset)
    
    common = []
    with open('intersection.pkl', 'rb') as f:
        common = pickle.load(f)

    for k in list(featuresDict.keys()):
        if k not in common:
            # pop out common feature
            featuresDict.pop(k)

    
    ##################################################################
    # featuresToAvoid = ['volt05', 'volt10', 'volt20', 'burstBandPowers','hFD']
    featuresList = list(featuresDict.keys())
    print(featuresList)

    featureMatrix = np.empty((0,ans.shape[1])) #[14*32 + 1,80640]
    for key,value in featuresDict.items():
        featureMatrix = np.append(featureMatrix,value,axis=0)


    if np.isnan(featureMatrix).any():
        featureMatrix = np.nan_to_num(featureMatrix,nan=0)

    featureMatrix = featureMatrix.astype('float64')


    feature_channel_index = []
    for feature in featuresList:
        for i in range(featuresDict[feature].shape[0]):
            if(i>=10):
                feature_channel_index.append(feature + str(i))
            else:
                feature_channel_index.append(feature + '0' + str(i))

    print(len(list(featuresDict.keys())))
    print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

    X = pd.DataFrame(featureMatrix.T)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X.columns = feature_channel_index

    #Remove Variance = 0 features     
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X)
    constant_columns = [column for column in X.columns
                    if column not in
    X.columns[constant_filter.get_support()]]
    X = constant_filter.transform(X)
    for column in constant_columns:
        feature_channel_index.remove(column)

    print(len(feature_channel_index),feature_channel_index )

    X = pd.DataFrame(X)
    X.columns = feature_channel_index

    #################################################################
    y = copy.deepcopy(Y_epoch[:,label]) #valence
    print("y.shape: ", y.shape)
    
    
    dfscores = None

    if(method == 'RandomForest'):
        '''Random Forest Feature Importances'''
        estimator = sklearnrfi() #RandomForestRegressor()
        fit = estimator.fit(X,y)
        dfscores = pd.DataFrame(fit.feature_importances_)
    elif(method == 'RFE'):
        ''' RFE'''
        selector = RFE(clf, n_features_to_select=X.shape[1], step=1)
        selector = selector.fit(X, y)
        dfscores = pd.DataFrame(selector.ranking_)

    elif(method == 'SelectKBest'):
        """SelecKBest"""
        #apply SelectKBest class to extract top 10 best features
        func = None
        if mutual_info == False:
            func = f_classif
        else:
            func = mutual_info_classif

        bestfeatures = SelectKBest(score_func=func, k=X.shape[1])
        fit = bestfeatures.fit(X,y)

        dfscores = pd.DataFrame(fit.scores_)

    
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    features_result = featureScores.nlargest(X.shape[1],'Score')
    print(features_result)
    features_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "CommonFeatureFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")



    ###################################################################
    topcolumns = features_result['Specs'].values
    topfeatures = []
    topelectrodes = []

    for col in topcolumns:
        feature = col[:-2]
        electrode = int(col[-2:])
        if(feature not in topfeatures):
            topfeatures.append(feature)
        
        if(electrode not in topelectrodes):
            topelectrodes.append(electrode)
    
    
    ######################################################################
    # TOP-N-FEATURE-RANKING
    print(topfeatures)
    print(topelectrodes)
    N =  len(topfeatures)
    topNRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]


    
    for n in range(1,N+1):
        
        topnfeatures = topfeatures[:n]
        
        # X-Values################################################

        featureMatrix = np.empty((0,ans.shape[1]))
    
        for feature in topnfeatures:
            featureMatrix = np.append(featureMatrix, featuresDict[feature], axis=0)
        
        featureMatrix = featureMatrix.astype('float64')
        print(featureMatrix.T.shape)

        feature_channel_index = []
        for feature in topnfeatures:
            i=0
            for i in range(featuresDict[feature].shape[0]):
                feature_channel_index.append(feature + str(i))

        
        # Removing NaN Values
        if np.isnan(featureMatrix).any():
            featureMatrix = np.nan_to_num(featureMatrix,nan=0)

        
        print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

        X = pd.DataFrame(featureMatrix.T)
        X.columns = feature_channel_index
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        

        print("Features Ready for undergoing selection tests done ...\n")

        X = X.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict,squared=False)
        print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        topNRmseList.append(rmse)



    
    topNFeatures_df = pd.DataFrame(topNList)

    topNRmse_df = pd.DataFrame(topNRmseList)

    #concat two dataframes for better visualization 
    topNFeaturesRanking = pd.concat([topNFeatures_df, topNRmse_df],axis=1)
    topNFeaturesRanking.columns = ['Feature','RMSE']  #naming the dataframe columns
    print(topNFeaturesRanking)
    topNFeaturesRanking.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topCommonFeatureFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")
    
    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(25, 10)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('Top N Features')
    plt.ylabel('RMSE')
    plt.plot(topNFeaturesRanking.loc[:,"Feature"], topNFeaturesRanking.loc[:,"RMSE"])
    plt.tight_layout()


# In[ ]:


def topFSColumnsRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest'):
        # Method C
        # parameters :-
                # dataset - name of the dataset
                # window - length of the sliding window in seconds
                # stride - length of the stride of the sliding window in seconds
                # sfreq - sampling frequency of the EEG data
                # clf - name of the classifier to be used
                # label - valence/arousal/dominance/liking label (shape depends upon the dataset)
                # scale - sclaing of the EEG data if required
                # mutual_info - Mutual ranking between features based on information theory
                # method - 'RandomForest' 'RFE' 'SelectKBest'
                
    # returns :-
                # void
    fs = sfreq
    pwd = os.getcwd()

    featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'
    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']

    print("Number of segments are: {}".format(ans.shape[1]))
    
    #X##############################################################################################
    
    featuresDict = None
    featuresDict = loadFeaturesDict(dataset)

    common = []
    with open('intersection.pkl', 'rb') as f:
        common = pickle.load(f)

    for k in list(featuresDict.keys()):
        if k not in common:
            # pop out common feature
            featuresDict.pop(k)

    print("Number of Features:",len(list(featuresDict.keys())))
    featuresList = list(featuresDict.keys())

    feature_channel_index = []

    feature_channel_index = []
    for feature in featuresList:
        for i in range(featuresDict[feature].shape[0]):
            if(i>=10):
                feature_channel_index.append(feature +'_'+  str(i))
            else:
                feature_channel_index.append(feature + '_0' + str(i))

    print(len(list(featuresDict.keys())))
    print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

    #defining feature matrix
    featureMatrix = np.empty((0,ans.shape[1])) #[14*32 + 1,80640]
    for key,value in featuresDict.items():
        featureMatrix = np.append(featureMatrix,value,axis=0)
        

    
    print("Shape of FeatureMatrix: {}\n".format(featureMatrix.T.shape))
    
    #data-imputation and nan-removal
    featureMatrix = featureMatrix.astype(np.float32)
    
    if np.isnan(featureMatrix).any():
        featureMatrix = np.nan_to_num(featureMatrix,nan=0)

    X = pd.DataFrame(featureMatrix.T)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X.columns = feature_channel_index
    

    #Y#####################################################################

    y = Y_epoch[:,label] #valence
    # y = pd.DataFrame(y)

    ########################################################################
    dfscores = None

    if(method == 'RandomForest'):
        '''Random Forest Feature Importances'''
        estimator = sklearnrfi() #RandomForestRegressor()
        fit = estimator.fit(X,y)
        dfscores = pd.DataFrame(fit.feature_importances_)
    elif(method == 'RFE'):
        ''' RFE'''
        selector = RFE(clf, n_features_to_select=X.shape[1], step=1)
        selector = selector.fit(X, y)
        dfscores = pd.DataFrame(selector.ranking_)

    elif(method == 'SelectKBest'):
        """SelecKBest"""
        #apply SelectKBest class to extract top 10 best features
        func = None
        if mutual_info == False:
            func = f_classif
        else:
            func = mutual_info_classif

        bestfeatures = SelectKBest(score_func=func, k=X.shape[1])
        fit = bestfeatures.fit(X,y)

        dfscores = pd.DataFrame(fit.scores_)
        



    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Column','Score']  #naming the dataframe columns
    features_result = featureScores.nlargest(X.shape[1],'Score')
    print(features_result)

    N = len(feature_channel_index)
    topNRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]

    for n in range(1, N+1):
        ranking_df = features_result.head(n)
        topncols = ranking_df['Column'].tolist()
        
        input_df = pd.DataFrame(X[topncols])

        X_train, X_test, y_train, y_test = train_test_split(input_df, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Apply classfier       
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict, squared=False)
        print(n,rmse)
        topNRmseList.append(rmse)


    topcol_df = pd.DataFrame(topNList)
    toprmse_df = pd.DataFrame(topNRmseList)
    #concat two dataframes for better visualization 
    topcolRanking = pd.concat([topcol_df, toprmse_df],axis=1)
    topcolRanking.columns = ['Column','RMSE']  #naming the dataframe columns
    topfeatures_result = topcolRanking
    print(topfeatures_result)
    topfeatures_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topFSColumnsRegressionRanking"+method + str(window) + str(stride) + ".csv")

    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(60, 9)

    plt.xlabel('Top N Columns')
    plt.ylabel('RMSE')
    plt.title("Top N Columns v/s RMSE Plot for Window:{} Stride:{} epoched data by varying N".format(window,stride))
    plt.plot(topfeatures_result.loc[:,"Column"], topfeatures_result.loc[:,"RMSE"])
    plt.tight_layout()
    plt.savefig(pwd + "/" + dataset + "/arousal_plots/" + "topFSColumnsRegressionRanking"+method + str(window) + str(stride) + ".svg", bbox_inches='tight', dpi=500)
    plt.show()
    plt.clf()


# In[ ]:


if __name__ == '__main__':
    pass
    

