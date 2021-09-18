#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone -l -s https://github.com/sari-saba-sadiya/EEGExtract.git cloned-repo')
get_ipython().run_line_magic('cd', 'cloned-repo')
get_ipython().system('ls')


# In[ ]:


get_ipython().system('pip  install -r requirements.txt')


# In[ ]:


from google.colab import drive
drive.mount('/gdrive',force_remount=True)


# In[ ]:


get_ipython().system('pip install pyinform')


# In[ ]:


get_ipython().run_line_magic('cd', '../../gdrive/MyDrive/emotion_recognition_project')


# In[ ]:


import EEGExtract as eeg
from scipy import io,signal
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# In[ ]:


class load_data:
    '''
    Load the preprocessed data here, store the paramters
    '''
    def __init__(self,name):
        self.name = name #name of dataset
        self.X = None
        self.Y = None
        self.Z = None
        self.freq = None #(in Hz) is same for all datasets
        self.channels = None
        self.ch_type = 'eeg'
        self.eegData = None
        self.use_autoreject = 'y'
        self.no_of_subjects = None
    def load_arrays(self):
        if self.name == 'DREAMER':
            array = np.load('original_data/DREAMER.npz')
            self.freq = 128
            self.no_of_subjects = 23
            self.channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
        if self.name == 'DEAP':
            array = np.load('original_data/DEAP.npz')
            self.no_of_subjects = 32
            self.freq = 128
            #                  0     1      2    3      4      5      6    7      8      9     10     11    12    13   14     15    16     17     18    19   20      21     22    23    24    25    26     27     28    29    30     31      32    33     34       35    36       37                 38                  39               
            self.channels = ['F1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'hEOG','vEOG', 'zEMG','tEMG','GSR','Respiration belt','Plethysmograph','Temperature'] 
        if self.name == 'OASIS':
            #array = np.load('original_data/OASIS.npz')
            self.no_of_subjects = 15
            if self.use_autoreject == 'y':
                with open('preprocessed_data/oasis/with_autoreject.p','rb') as file:
                    self.X = pickle.load(file)
                    self.channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
                    self.freq = 128
                    self.X ,self.Y= merge_dictionary(self.X)
                    (a,b,c) = self.X.shape
                    self.X = np.reshape(self.X,(a,c,b))
            else:
                array = np.load('preprocessed_data/oasis/without_autoreject.npz')
                self.freq = 128
                self.channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
                self.X = array['X']
                self.Y = array['Y']
                (a,b,c) = self.X.shape
                self.X = np.reshape(self.X,(a,c,b))

        else:
            self.X = array['X']
        if self.name == 'DEAP':
            self.X = self.X[:,:,[1,3,2,4,7,11,13,31,29,25,21,19,20,17]] # To maintain uniformity across all datasets, only 14 electrodes selected
            self.channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        if self.name != 'OASIS':  
              self.Y = array['Y']
              #self.Z = array['Z']
        self.reshape_data()
    def reshape_data(self):
      '''
      reshapes data in the format EEGExtract module expects i.e channels x timepoints x epochs 
      '''

        (epochs,timepoints,channels) = self.X.shape
        self.eegData = np.reshape(self.X,(channels,timepoints,epochs)) 
  
        


# In[ ]:


def merge_dictionary(dictionary):
  '''
  merge all trial data to form one array
  '''
    no_of_trials = len(list(dictionary.keys()))
    no_of_channels = dictionary[1][0].shape[1]
    length_of_segment = dictionary[1][0].shape[2]
    no_of_epochs_per_trial = dictionary[1][0].shape[0]
    X = np.empty((0,no_of_channels,length_of_segment))
    Y = np.empty((0,2))
    for trial,lst in dictionary.items():
        array = dictionary[trial][0]
        score = dictionary[trial][3]
        X = np.append(X,array,axis = 0)
    for epoch in range(no_of_epochs_per_trial):
        Y = np.append(Y,np.expand_dims(score,axis =0),axis = 0)
    
    return X,Y


# In[ ]:


def calculate_diffrential_entropy_for_bands(eegData,freq):
# Function to calculate the differential entropy for the different bands of EEG data

# parameters :-
            # eegData :- The differential EEG signal value
            # freq :- sampling frequency of the EEG signal
# returns :-
            # bandwise DE
  #delta band
    delta_band = eeg.filt_data(eegData,0.5,4,freq)
  #theta band
    theta_band = eeg.filt_data(eegData,4,8,freq)
  #alpha bad
    alpha_band = eeg.filt_data(eegData,8,12,freq)
  #beta band
    beta_band = eeg.filt_data(eegData,12,30,freq)
  #gamma band
    gamma_band = eeg.filt_data(eegData,30,63,freq)


    diffrential_entropy_delta = 1/2*np.log(np.var(delta_band,axis = 1)*np.pi*np.e*2)
  
    diffrential_entropy_theta = 1/2*np.log(np.var(theta_band,axis = 1)*np.pi*np.e*2)
  
    diffrential_entropy_alpha = 1/2*np.log(np.var(alpha_band,axis = 1)*np.pi*np.e*2)
  
    diffrential_entropy_beta = 1/2*np.log(np.var(beta_band,axis = 1)*np.pi*np.e*2)
  
    diffrential_entropy_gamma = 1/2*np.log(np.var(gamma_band,axis = 1)*np.pi*np.e*2)
  #print(diffrential_entropy_delta.shape,diffrential_entropy_gamma.shape,diffrential_entropy_theta.shape,diffrential_entropy_alpha.shape,diffrential_entropy_beta.shape)
    return diffrential_entropy_delta,diffrential_entropy_theta,diffrential_entropy_alpha,diffrential_entropy_beta,diffrential_entropy_gamma


# In[ ]:


#['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
#   0      1     2     3      4    5     6     7     8     9      10    11     12    13
def calculate_RASM_DASM(band):
    RASM_AF3_AF4 = np.expand_dims(band[0,:]/band[13,:],axis = 0)
    RASM_F3_F4 = np.expand_dims(band[2,:]/band[11,:],axis = 0)
    RASM_F7_F8 = np.expand_dims(band[1,:]/band[12,:],axis = 0)
    RASM_FC5_FC6 = np.expand_dims(band[3,:]/band[10,:],axis = 0)
    RASM_O1_O2 = np.expand_dims(band[6,:]/band[7,:],axis = 0)
    RASM_P7_P8 = np.expand_dims(band[5,:]/band[8,:],axis=0)
    RASM_T7_T8 = np.expand_dims(band[4,:]/band[9,:],axis=0)

    DASM_AF3_AF4 = np.expand_dims(band[0,:]-band[13,:],axis = 0)
    DASM_F3_F4 = np.expand_dims(band[2,:]-band[11,:],axis = 0)
    DASM_F7_F8 = np.expand_dims(band[1,:]-band[12,:],axis = 0)
    DASM_FC5_FC6 = np.expand_dims(band[3,:]-band[10,:],axis = 0)
    DASM_O1_O2 = np.expand_dims(band[6,:]-band[7,:],axis = 0)
    DASM_P7_P8 = np.expand_dims(band[5,:]-band[8,:],axis=0)
    DASM_T7_T8 = np.expand_dims(band[4,:]-band[9,:],axis=0)

  
    features = np.empty((0,RASM_AF3_AF4.shape[1]))
    features = np.append(features,RASM_AF3_AF4,axis = 0)
    features = np.append(features,RASM_F3_F4,axis = 0)
    features = np.append(features,RASM_F7_F8,axis = 0)
    features = np.append(features,RASM_FC5_FC6,axis = 0)
    features = np.append(features,RASM_O1_O2,axis = 0)
    features = np.append(features,RASM_P7_P8,axis = 0)
    features = np.append(features,RASM_T7_T8,axis = 0)

    features = np.append(features,DASM_AF3_AF4,axis = 0)
    features = np.append(features,DASM_F3_F4,axis = 0)
    features = np.append(features,DASM_F7_F8,axis = 0)
    features = np.append(features,DASM_FC5_FC6,axis = 0)
    features = np.append(features,DASM_O1_O2,axis = 0)
    features = np.append(features,DASM_P7_P8,axis = 0)
    features = np.append(features,DASM_T7_T8,axis = 0)
    return features.T


# In[ ]:


def epoch_data(X,Y, window, stride, sfreq):

    (channels,timepoints,trials )= X.shape
    X = np.reshape(X,(trials,channels,timepoints)) 
    segment = int(window*sfreq)
    step = int(stride*sfreq)
    epochPerTrial = int((timepoints-segment)/step + 1)
    count = 0
    X_new = np.empty((trials*epochPerTrial,channels,segment))
    Y_new = np.empty((trials*epochPerTrial,2))
    for trial in range(trials):
        for epoch in range(epochPerTrial):
            X_new[count,:,:] = X[trial,:,epoch*step:(epoch*step)+segment]
            Y_new[count,:] = Y[trial,:2]
            count+=1
    (trials,channels,timepoints) = X_new.shape
    X_new = np.reshape(X_new,(channels,timepoints,trials))

    return X_new,Y_new


# In[ ]:


def segregate_data_of_subjects(feature_matrix,dataset,sfreq = 128):
    total_samples = feature_matrix.shape[0]
    subject_indexes = {}
    if dataset.name != 'DEAP AND DREAMER':
        samples_per_subject = total_samples//dataset.no_of_subjects
        print('samples per subject taken are ',samples_per_subject)
        subject_indexes = {}
        for i in range(dataset.no_of_subjects):
            subject_name = 'subject_' + str(i+1)
            subject_indexes[subject_name] = feature_matrix[samples_per_subject*i:samples_per_subject*(i+1),:]
    else:
        a = feature_matrix[:80640,:]
        b = feature_matrix[80640:,:]
        print(b.shape)
        for i in range(32):
            samples_per_subject = 2520
            subject_name = 'subject_' + str(i+1)
            subject_indexes[subject_name] = a[samples_per_subject*i:samples_per_subject*(i+1),:]
        for i in range(0,23):
            samples_per_subject = 8190
            subject_name = 'subject_' + str(i+1+32)
            subject_indexes[subject_name] = b[samples_per_subject*i:samples_per_subject*(i+1),:]

    return subject_indexes


# In[ ]:


def driver_code():
    dataset = load_data('DREAMER')
    dataset.load_arrays()
    X = dataset.eegData
    Y = dataset.Y 
    window = 1
    stride = 1
  
    X,Y = epoch_data(X,Y,window,stride,128)
    print('shape after epoching')
    print('X:',X.shape)
    print('Y:',Y.shape)
    print('')
    print('')
    delta,theta,alpha,beta,gamma = calculate_diffrential_entropy_for_bands(X,dataset.freq)
    bands = {'delta':delta,'theta':theta,'alpha':alpha,'beta':beta,'gamma':gamma}
    for name,band in bands.items():
        feature_matrix = calculate_RASM_DASM(band) #extracted RASM ,DASM features for each eng band
        print(name ,':' ,end = '')
        print(feature_matrix.shape)
        print(feature_matrix)
        np.savez('features/'+dataset.name.lower()+'_RASM_DASM/'+name+'_'+str(window)+'_'+str(stride),features = feature_matrix,Y=Y)



# In[ ]:


driver_code()


# In[ ]:



np.load('features/oasis/without_autoreject/shannonEntropy_1_1.npz')['features']


# In[ ]:




