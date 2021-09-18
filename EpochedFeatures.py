#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ImportUtils
import math 
import EEGExtract as eeg
from sklearn.model_selection import train_test_split

import os
import glob
from scipy import io,signal
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os


# In[ ]:


def select_channels(data,channels):
    
# parameters:- 
            # data - channelwise EEG preprocessed data
            # channels - list of required channels
        
# returns:-
            # ans - the selected channels from the entire dataset
                
    ans = np.empty((data.shape[0],len(channels),data.shape[2]))
    for sub in range(data.shape[0]):
        ans[sub,:,:] = np.array([data[sub,x,:] for x in channels])
    return ans


# In[ ]:


def epoch_data(X, Y, Z, window, stride, sfreq):

# Function to epoch the data

# parameters:-
            # X - The EEG data input passed as trial*channel*timepoints
            # Y - VALD (depending on the dataset) as given by the user
            # Z - Participant number and session number
            # window - length of required window in seconds
            # stride - stride of the required sliding window in seconds
            # sfreq - sampling frequency of the obtained EEG data
            
# retruns:-
            # X_new - Epoched X
            # Y_new - Epoched Y (All the segments for a given trial shall have the same value, i.e the one given by the subject)
            # Z_new - Epoched Z (All the segments for a given trial shall have the same value, i.e the one of the subject)
            
    trials,channels,timepoints = X.shape
    segment = int(window*sfreq)
    step = int(stride*sfreq)
    epochPerTrial = int((timepoints-segment)/step + 1)

    X_new = np.empty((trials*epochPerTrial,channels,segment))
    Y_new = np.empty((trials*epochPerTrial,Y.shape[1]))
    Z_new = np.empty((trials*epochPerTrial,Z.shape[1]))

    count=0
    for trial in range(trials):
        for epoch in range(epochPerTrial):
            X_new[count,:,:] = X[trial,:,epoch*step:(epoch*step)+segment]
            Y_new[count,:] = Y[trial,:]
            Z_new[count,:] = Z[trial,:]
            count = count+1
            
    return X_new, Y_new, Z_new


# In[ ]:


def save_features(dataset, ans, Y_epoch, sfreq, window, stride):

# A function to generate the features and save them
    
# parameters:-
            # dataset - name of the dataset
            # ans - epoched segment of X
            # Y_epoch - epoched segments of the valence and arousal scores taken from the subject
            # window - window length in seconds
            # stride - stride length in seconds
            # sfreq - sampli5ng frequency of the EEG signal
            
# returns:-
            # void
    
    fs = sfreq
    
    featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'
    
    feature_matrix = eeg.shannonEntropy(ans, bin_min=-200, bin_max=200, binWidth=2)
    np.savez((featurepath+"shannonEntropy_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.eegStd(ans)
    stdshape = feature_matrix.shape
    # The channels 
    emotiv_channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    left_channels = ['AF3', 'F7','F3', 'FC5', 'T7', 'P7', 'O1']
    right_channels = ['AF4','F8','F4','FC6','T8','P8','O2']

    dasm_gamma = np.empty((0,stdshape[1]))
    rasm_gamma = np.empty((0,stdshape[1]))
    for lc,rc in zip(left_channels, right_channels):
        lci = emotiv_channels.index(lc)
        rci = emotiv_channels.index(rc)
        
        #left differential entropy
        dl = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[lci,:,:], axis=0),30,45,fs)))))
        #right differential entropy
        dr = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[rci,:,:], axis=0),30,45,fs)))))

        dasm_gamma = np.append(dasm_gamma, np.subtract(dl,dr), axis=0)
        rasm_gamma = np.append(rasm_gamma, np.divide(dl,dr), axis=0)

    np.savez((featurepath+"dasm_gamma_{}_{}.npz").format(window,stride),features = dasm_gamma , Y = Y_epoch)
    np.savez((featurepath+"rasm_gamma_{}_{}.npz").format(window,stride),features = rasm_gamma , Y = Y_epoch)
    del dasm_gamma, rasm_gamma

    return 
    '''
    Subband Information Quantity
    '''
    
    # delta (0.5–4 Hz)
    eegData_delta = eeg.filt_data(ans, 0.5, 4, fs)
    feature_matrix = eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2)
    np.savez((featurepath+"ShannonRes_sub_bands_delta_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    eegData_theta = eeg.filt_data(ans, 4, 8, fs)
    feature_matrix = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
    np.savez((featurepath+"ShannonRes_sub_bands_theta_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    eegData_alpha = eeg.filt_data(ans, 8, 12, fs)
    feature_matrix = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
    np.savez((featurepath+"ShannonRes_sub_bands_alpha_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)
    
    eegData_beta = eeg.filt_data(ans, 12, 30, fs)
    feature_matrix = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
    np.savez((featurepath+"ShannonRes_sub_bands_beta_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    eegData_gamma = eeg.filt_data(ans, 30,45, fs)
    feature_matrix = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)
    np.savez((featurepath+"ShannonRes_sub_bands_gamma_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    HjorthMob, HjorthComp = eeg.hjorthParameters(ans)
    feature_matrix = HjorthComp
    np.savez((featurepath+"Hjorth_complexity_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = HjorthMob
    np.savez((featurepath+"Hjorth_mobilty_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)
    
    feature_matrix = eeg.falseNearestNeighbor(ans)
    np.savez((featurepath+"falseNearestNeighbor_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.medianFreq(ans,fs)
    np.savez((featurepath+"medianFreq_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.bandPower(ans, 0.5, 4, fs)
    np.savez((featurepath+"bandPwr_delta_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.bandPower(ans, 4, 8, fs)
    np.savez((featurepath+"bandPwr_theta_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.bandPower(ans, 8, 12, fs)
    np.savez((featurepath+"bandPwr_alpha_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.bandPower(ans, 12, 30, fs)
    np.savez((featurepath+"bandPwr_beta_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.bandPower(ans, 30, 45, fs)
    np.savez((featurepath+"bandPwr_gamma_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)
      
    feature_matrix = eeg.eegStd(ans)
    stdshape = feature_matrix.shape
    np.savez((featurepath+"stdDev_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.diffuseSlowing(ans)
    np.savez((featurepath+"diffuseSlowing_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    minNumSamples = int(70*fs/1000)
    feature_matrix = eeg.spikeNum(ans,minNumSamples)
    np.savez((featurepath+"spikeNum_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    
    feature_matrix = eeg.burstAfterSpike(ans,eegData_delta,minNumSamples=7,stdAway = 3)
    np.savez((featurepath+"deltaBurstAfterSpike_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.shortSpikeNum(ans,minNumSamples)
    np.savez((featurepath+"shortSpikeNum_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.numBursts(ans,fs)
    np.savez((featurepath+"numBursts_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    burstLenMean_res,burstLenStd_res = eeg.burstLengthStats(ans,fs)
    feature_matrix = burstLenMean_res 
    np.savez((featurepath+"burstLen_u_and_sigma_mean_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = burstLenStd_res
    np.savez((featurepath+"burstLen_u_and_sigma_std_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    feature_matrix = eeg.numSuppressions(ans,fs)
    np.savez((featurepath+"numSuppressions_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)

    suppLenMean_res,suppLenStd_res = eeg.suppressionLengthStats(ans,fs)
    feature_matrix = suppLenMean_res
    np.savez((featurepath+"suppressionLen_u_and_sigma_mean_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)
    del suppLenMean_res

    feature_matrix = suppLenStd_res
    np.savez((featurepath+"suppressionLen_u_and_sigma_std_{}_{}.npz").format(window,stride),features = feature_matrix , Y = Y_epoch)
    del suppLenStd_res

    # DASM and RASM Features
    # DASM = h(X lefti) − h(Xrighti), and (2)
    # RASM = h(Xlefti)/h(Xrighti),

    emotiv_channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    left_channels = ['AF3', 'F7','F3', 'FC5', 'T7', 'P7', 'O1']
    right_channels = ['AF4','F8','F4','FC6','T8','P8','O2']

    #[chans x ms x epochs] 
    dasm_delta = np.empty((0,stdshape[1]))
    rasm_delta = np.empty((0,stdshape[1]))
    for lc,rc in zip(left_channels, right_channels):
        lci = emotiv_channels.index(lc)
        rci = emotiv_channels.index(rc)
        
        #left differential entropy
        inputarr = np.expand_dims(ans[lci,:,:], axis=0)
        print("inputarr.shape=", inputarr.shape)
        temp = eeg.filt_data(inputarr, 0.5, 4, fs)
        tempstd = eeg.eegStd(temp)
        

        dl = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[lci,:,:], axis=0), 0.5, 4, fs)))))
        #right differential entropy
        dr = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[rci,:,:], axis=0), 0.5, 4, fs)))))
        
        print("temp.shape=", temp.shape,"tempstd.shape=", tempstd.shape,"dl.shape= ", dl.shape, "stdshape=", stdshape)
        dasm_delta = np.append(dasm_delta, np.subtract(dl,dr), axis=0)
        rasm_delta = np.append(rasm_delta, np.divide(dl,dr), axis=0)
    
    np.savez((featurepath+"dasm_delta_{}_{}.npz").format(window,stride),features = dasm_delta , Y = Y_epoch)
    np.savez((featurepath+"rasm_delta_{}_{}.npz").format(window,stride),features = rasm_delta , Y = Y_epoch)
    del dasm_delta, rasm_delta

    dasm_theta = np.empty((0,stdshape[1]))
    rasm_theta = np.empty((0,stdshape[1]))
    for lc,rc in zip(left_channels, right_channels):
        lci = emotiv_channels.index(lc)
        rci = emotiv_channels.index(rc)
        
        #left differential entropy
        dl = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[lci,:,:], axis=0), 4, 8, fs)))))
        #right differential entropy
        dr = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[rci,:,:], axis=0), 4, 8, fs)))))

        dasm_theta = np.append(dasm_theta, np.subtract(dl,dr), axis=0)
        rasm_theta = np.append(rasm_theta, np.divide(dl,dr), axis=0)

    np.savez((featurepath+"dasm_theta_{}_{}.npz").format(window,stride),features = dasm_theta , Y = Y_epoch)
    np.savez((featurepath+"rasm_theta_{}_{}.npz").format(window,stride),features = rasm_theta , Y = Y_epoch)
    del dasm_theta, rasm_theta

    dasm_alpha = np.empty((0,stdshape[1]))
    rasm_alpha = np.empty((0,stdshape[1]))
    for lc,rc in zip(left_channels, right_channels):
        lci = emotiv_channels.index(lc)
        rci = emotiv_channels.index(rc)
        
        #left differential entropy
        dl = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[lci,:,:], axis=0), 8, 12, fs)))))
        #right differential entropy
        dr = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[rci,:,:], axis=0), 8, 12, fs)))))

        dasm_alpha = np.append(dasm_alpha, np.subtract(dl,dr), axis=0)
        rasm_alpha = np.append(rasm_alpha, np.divide(dl,dr), axis=0)

    np.savez((featurepath+"dasm_alpha_{}_{}.npz").format(window,stride),features = dasm_alpha , Y = Y_epoch)
    np.savez((featurepath+"rasm_alpha_{}_{}.npz").format(window,stride),features = rasm_alpha , Y = Y_epoch)
    del dasm_alpha, rasm_alpha

    
    dasm_beta = np.empty((0,stdshape[1]))
    rasm_beta = np.empty((0,stdshape[1]))
    for lc,rc in zip(left_channels, right_channels):
        lci = emotiv_channels.index(lc)
        rci = emotiv_channels.index(rc)
        
        #left differential entropy
        dl = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[lci,:,:], axis=0), 12, 30,fs)))))
        #right differential entropy
        dr = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[rci,:,:], axis=0), 12, 30,fs)))))

        dasm_beta = np.append(dasm_beta, np.subtract(dl,dr), axis=0)
        rasm_beta = np.append(rasm_beta, np.divide(dl,dr), axis=0)

    np.savez((featurepath+"dasm_beta_{}_{}.npz").format(window,stride),features = dasm_beta , Y = Y_epoch)
    np.savez((featurepath+"rasm_beta_{}_{}.npz").format(window,stride),features = rasm_beta , Y = Y_epoch)
    del dasm_beta, rasm_beta



    dasm_gamma = np.empty((0,stdshape[1]))
    rasm_gamma = np.empty((0,stdshape[1]))
    for lc,rc in zip(left_channels, right_channels):
        lci = emotiv_channels.index(lc)
        rci = emotiv_channels.index(rc)
        
        #left differential entropy
        dl = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[lci,:,:], axis=0),30,45,fs)))))
        #right differential entropy
        dr = (0.5)*np.log((2*math.pi*math.e*np.square(eeg.eegStd(eeg.filt_data(np.expand_dims(ans[rci,:,:], axis=0),30,45,fs)))))

        dasm_gamma = np.append(dasm_gamma, np.subtract(dl,dr), axis=0)
        rasm_gamma = np.append(rasm_gamma, np.divide(dl,dr), axis=0)

    np.savez((featurepath+"dasm_gamma_{}_{}.npz").format(window,stride),features = dasm_gamma , Y = Y_epoch)
    np.savez((featurepath+"rasm_gamma_{}_{}.npz").format(window,stride),features = rasm_gamma , Y = Y_epoch)
    del dasm_gamma, rasm_gamma


# In[ ]:


def getEpochedFeatures(dataset, window, stride, sfreq, label):
    
# Function to reshape the arrays before passing on to the save_features function

# parameters:-
            # dataset - name of the dataset
            # window - length of window in seconds
            # stride - length of stride in seconds
            # sfreq - sampling frequency of the EEG signal
            # label - valence/arousal (0/1)
            
# returns:-
            # void
    '''
    Returns Accuracy vs Segment size plot for
    window - length of window
    stride - step 
    sfreq - sampling freq
    label - 0-valence, 1-arousal, 2-dominance, 3-liking
    '''
    fs = sfreq
    X = None
    Y = None
    Z = None
    pwd = os.getcwd()
    with np.load((pwd + '/data_extracted/{}.npz').format(dataset), allow_pickle=True) as data:
        X = data['X']
        Y = data['Y']
        Z = data['Z']

    print("Shape After Loading")
    print("X.shape=", X.shape," Y.shape=",Y.shape," Z.shape=", Z.shape)
    # return 
    #########!MODIFY FOR DREAMER AND DEAP DATASET########################################
    #****
    '''
    Reshape Data
    '''
    if(dataset != "DEAP"):
        temp_arr = np.empty((X.shape[0],X.shape[2],X.shape[1]))
        for i in range(temp_arr.shape[0]):
            temp_arr[i,:,:] = X[i,:,:].transpose()
        X = copy.deepcopy(temp_arr)
        del temp_arr

    print("Shape after reshaping")
    print("X.shape=", X.shape," Y.shape=",Y.shape," Z.shape=", Z.shape)
    '''
    Select Channels(if needed)
    '''
    
    print("Data Loaded...\n")
    ch_names = ['F1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'hEOG','vEOG', 'zEMG','tEMG','GSR','Respiration belt','Plethysmograph','Temperature']
    emotiv_channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    index_arr = [ch_names.index(x) for x in emotiv_channels]

    X_new = None
    if(dataset == "DEAP"):
        X_new = select_channels(X,index_arr)
    else:
        X_new = copy.deepcopy(X)
    
    print("X_new.shape = ", X_new.shape)
    
    del X
    print("Channel selection done ...\n")
    '''
    #  X = (32*40,40,8064)
    #  Y = (32*40,4)
    #  Z = (32*40,2)

    # X :  (nbSegments, nbChannel, nbTimepoints) : Data
    # Y :  (nbSegments, nbEmotions) : Valence and arousal data
    # Z :  (nbSegments, 2) : Participant number, and session number
    '''

    '''
    DREAMER Dataset
    #         X = (23*18,7808+54032,14)
    #         Y = (23*18,2)
    #         Z = (23*18,2)
    '''
    
    (X_epoch, Y_epoch, Z_epoch) = epoch_data(X_new, Y, Z,window,stride,sfreq)
    del X_new
    del Y
    del Z

    print("Epoching done ...\n")
    print(X_epoch.shape, Y_epoch.shape, Z_epoch.shape) #debug

    # 1280*63,40,128
    # trial, channel, segment
    trials, channels, segment = X_epoch.shape
    ans = np.empty((channels, segment, trials)) #[chans x ms x epochs] 
    for i in range(trials):
        ans[:,:,i] = X_epoch[i,:,]
    del X_epoch

    print("ans.shape = ", ans.shape)
    print("Rotation of np.array done ...\n")
    pwd = os.getcwd()
    filepath = pwd + '/' + dataset + "/data_extracted/epochedData/data" + str(window) + str(stride) + ".npz"
    np.savez(filepath,ans,Y_epoch, Z_epoch)

    # featuresDict = getFeaturesDict(ans,sfreq)
    save_features(dataset, ans, Y_epoch, sfreq, window, stride)

    # with open(pwd + '/' + dataset + '/data_extracted/featureDicts/'+str(window)+str(stride)+ '.pkl', 'wb') as f:
    #     pickle.dump(featuresDict, f, pickle.HIGHEST_PROTOCOL)

    print("Feature Extraction done ...\n")
    

if __name__ == '__main__':
    pass

