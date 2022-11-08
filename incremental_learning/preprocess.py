#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive 
drive.mount('/gdrive',force_remount = True)


# In[ ]:


get_ipython().system('pip install mne')


# In[ ]:


get_ipython().system('pip install autoreject')


# In[ ]:


import numpy as np
import mne
import autoreject 
from scipy.stats import pearsonr
import pickle


# In[ ]:


get_ipython().run_line_magic('cd', '/gdrive/MyDrive/emotion_recognition_project/')


# In[ ]:


class preprocessing:
    '''
    Load the data here, store the paramters
    '''
    def __init__(self,name):
        self.name = name #name of dataset
        self.X = None
        self.Y = None
        self.Z = None
        self.gyroscope = None
        self.freq = None #(in Hz) is same for all datasets
        self.channels = None
        self.ch_type = 'eeg'
    def load_arrays(self):
        '''
          loads arrays in object variables of the form 
          X: trials x channels x timepoints, using reshape method at the end
          Y: trials x (valence,arousal)
          Z: trials x participant no
        '''
        if self.name == 'DREAMER':
            array = np.load('original_data/DREAMER.npz')
            self.freq = 128
            self.channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
        if self.name == 'DEAP':
            array = np.load('original_data/DEAP.npz')
            self.freq = 128
            self.channels = ['F1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'hEOG','vEOG', 'zEMG','tEMG','GSR','Respiration belt','Plethysmograph','Temperature'] 
        if self.name == 'OASIS':
            array = np.load('original_data/OASIS.npz')
            self.channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            self.freq = 128
        self.X = array['X']
        if self.name == 'DEAP':
            self.X = self.X[:,:,:32]
            self.channels = self.channels[:32]
        
        if self.name == 'OASIS':
            self.gyroscope = array['gyroscope']
        
        self.Y = array['Y']
        self.Z = array['Z']
        self.reshape_data()
            
    def reshape_data(self):
      '''
      exchanges last two dimensions of data
      '''
        (a,b,c) = self.X.shape 
        self.X = np.reshape(self.X,(a,c,b)) 
        
        


# In[ ]:


class filters():
  '''
  define filters to be used for preprocessing 
  '''
    @staticmethod
    def notch_filter(data,sfreq,notch_freq):
        # parameters :-
                    # data :- EEG data
                    # sfreq :- sampling frequency
                    # notch_freq :- frequency of the notch filter (generally 50Hz due to the AC current frequency)
        return  mne.filter.notch_filter(data,sfreq,np.arange(notch_freq,notch_freq+1,1))

    @staticmethod
    def butterworth_filter(data,sfreq,lfreq,hfreq):
        # parameters :-
                    # data :- EEG data
                    # sfreq :- sampling frequency
                    # lfreq :- low pass frequency value
                    #hfreq :- high pass frequency value
        return mne.filter.filter_data(data  = data,sfreq = sfreq,l_freq = lfreq,h_freq = hfreq,method = 'iir',verbose = False)
        


# In[ ]:


class referencing():
  '''
  referencing electrodes to some value
  '''
    @staticmethod
    def average(data):
    '''
    Computes average voltage  of all channels for a particular trial and a particular timepoint, and subtracts average value from all channels 
    '''
        temp = data
        avg = np.average(temp,axis=1)
        avg = np.expand_dims(avg,axis=1)
        return temp-avg


# In[ ]:


class autoreject_custom:
  '''
  Run Auotoreject algorithm here for artifact rejections
  '''
  #make epoch object
    @staticmethod
    def raw_object_creation(raw_data,channel_name,ch_types,sfreq):
    '''
    defining parameters for creation of raw object  which will be used for creating an epoch object
    retutns raw object after setting parameters
    '''
    # parameters :-
                # raw_data :- EEG data
                # channel_name :- Names of the channels of EEG data used
                # ch_types :- Whether each channel is EEG/Gyro, etc
                # sfreq :- sampling frequency
        montage = mne.channels.make_standard_montage('standard_1020')

    #creating a info object to create epochs later and setting its montage to 12-20 system
        info = mne.create_info(ch_names=channel_name,sfreq=sfreq,ch_types = ch_types,verbose = False)

    #create raw object directly from array
        raw_object = mne.io.RawArray(data = raw_data,info = info,verbose = False)

    #setting montage
        raw_object.set_montage(montage)

        return raw_object
  
    @staticmethod
    def epoch_object_creation(raw_object,start=0,duration=1,tmin=0,tmax=0.99):
    '''
    making an epoch object which will be used for autoreject algorithm
    '''
    #creating fixed length events
        events = mne.make_fixed_length_events(raw_object,id=1,start=0,duration = duration)
    #creating an epoch object
        epoch_object = mne.Epochs(raw_object,events = events,preload=True,baseline = None,reject=None,verbose=False,tmin=0,tmax=0.99)

        return epoch_object

    @staticmethod
    def autoreject_algo(epoch_object,n_interpolates,consensus_percs):
    '''
    cleans the epochs,and returns cleaned epochs,rejecting bad epochs based on optimal parameters calculation
    n_interpolates are the ρ values that we would like autoreject to try and consensus_percs are the κ values that autoreject will try
    Epochs with more than κ∗N sensors (N total sensors) bad are dropped
    '''
        ar = autoreject.AutoReject(n_interpolates, consensus_percs, random_state=42,verbose = 'tqdm_notebook',cv=4,n_jobs=10)
    #fitting autoreject model to epoch data
        ar.fit(epoch_object)
        epochs_clean = ar.transform(epoch_object)
        evoked_clean = epochs_clean.average()
        evoked = epoch_object.average()

        return epochs_clean,ar.get_reject_log(epoch_object)

      


# In[ ]:


class source_decomposition():

    @staticmethod
    def ica(data,channels,ch_type,sfreq):
    # parameters :-
                # data :- EEG data
                # channels :- Names of the channels of EEG data used
                # ch_types :- Whether each channel is EEG/Gyro, etc
                # sfreq :- sampling frequency        
    #defining ICA parameters
        raw = autoreject_custom.raw_object_creation(data,channels,ch_type,sfreq)
        ica = mne.preprocessing.ICA(method='infomax',n_components=14)
        ica.fit_params['max_iter'] =300
        ica.fit(raw,verbose=False)
        return ica.get_sources(raw).get_data(),ica.mixing_matrix_


# In[ ]:


def process_trial(a,acc_x,acc_y,acc_z):
  '''
  a are the source signals obtained after decomposition
  acc_<> are accelerometer readings in respective axis
  '''
# parameters :-
            # a :- EEG source signal after ICA
            # acc_x :- accelerometer channel along X axis
            # acc_y :- accelerometer channel along Y axis
            # acc_y :- accelerometer channel along Z axis
            
#pearson co-eff between each source signal,and accelerometer readings
    pcoeff_arr = np.zeros((a.shape[0],3))#array will record p_coeff for each source with x,y,z accelermeter readings
    for i in  range(a.shape[0]):
        source = a[i] #extracting particular source
        #calculating pearson co-relation coeff between particular source each of accelerometer axis readings 
        r_x,_ = pearsonr(source,acc_x) 
        r_y,_ = pearsonr(source,acc_y)
        r_z,_ = pearsonr(source,acc_z)
        pcoeff_arr[i,0] = r_x
        pcoeff_arr[i,1] = r_y
        pcoeff_arr[i,2] = r_z
    #print('############')
  #calculating mean ,std deviation of pearson co-eff for all sources for each axis i.e X,Y,Z
    mean = np.mean(pcoeff_arr,axis = 0)
    std = np.std(pcoeff_arr,axis = 0)
    error = mean + 2 * std

  #calculating which sources differ have pearson co-eff of atleast one axis greater than 2 standard deviation from mean
    bad_source_index = []
    for i in range(pcoeff_arr.shape[0]):
        if pcoeff_arr[i,0] > error[0] or pcoeff_arr[i,1] > error[1] or pcoeff_arr[i,2] > error[2]:
            bad_source_index.append(i)

  #correcting bad sources by butterworth filter by high pass 3Hz frequency as motion artifacts are said to exist in low power frequencies
    for index in bad_source_index:
        source_to_be_filtered = a[index]
        a[index] = filters.butterworth_filter(source_to_be_filtered,dataset.freq,3,None)#high pass filter 3Hz

    return a #return corrected source signals


# In[ ]:


#loading dataset arrays
dataset = preprocessing('OASIS')
dataset.load_arrays()
dataset.gyroscope.shape


# In[ ]:


#referencing electrodes  to average value method
average_data = referencing.average(dataset.X)


# In[ ]:


#running butterworth filter (bandpass filter)
filtered_data = filters.notch_filter(average_data,dataset.freq,60)#butterworth_filter(average_data,dataset.freq,0.1,40)


# In[ ]:


no_of_trials = dataset.X.shape[0]

(a,b,c) = dataset.gyroscope.shape
gyroscope_trials = np.reshape (dataset.gyroscope,(a,c,b))# reshaping trials so they are of the shape trials x channels x timepoints

#iterating over all trials and correcting trial data for motion artifact
for trial_n in range(no_of_trials):
    print('processing trial no:',trial_n+1)
    trial_data = filtered_data[trial_n]
    gyroscope_trial  = gyroscope_trials[0,4:,:] #only acclerometer values extracted for a particular trial
    gyroscope_trial_x = gyroscope_trial[0] # accelerometer x axis reading
    gyroscope_trial_y = gyroscope_trial[1] # accelerometer y axis reading
    gyroscope_trial_z = gyroscope_trial[2] # accelerometer z axis reading
    source_signals,mixing_matrix = source_decomposition.ica(trial_data,dataset.channels,dataset.ch_type,dataset.freq)
    corrected_sources = process_trial(source_signals,gyroscope_trial_x,gyroscope_trial_y,gyroscope_trial_z)

  #corrected sources are projected back into orignal dimensional space of EEG data using mixing matrix
    project_back = np.matmul(mixing_matrix,corrected_sources)
    filtered_data[trial_n] = project_back


# In[ ]:


filtered_data.shape


# In[ ]:


no_of_trials = dataset.X.shape[0]
'''
dictionary contains information about each trial
each trial number i is mapped to a list containing the cleaned epochs given by autoreject,boolena array indicating which epoch was dropped,and 
a percentage indicating epochs dropped out of total, valence ,arousal rating for  trial and image_id
'''
#running autoreject for each trial data

'''
autoreject divides each trial data into 5 epochs of 1 sec segment i.e 640 timepoints into 128 timepoints per epochs,and runs algo on each
epoch,rejecting epochs based on estimated parameters
'''
clean_epochs ={}
for trial in range(no_of_trials):
    print('trial no',trial)
    temp = filtered_data[trial]
    raw_object = autoreject_custom.raw_object_creation(temp,dataset.channels,dataset.ch_type,dataset.freq)
    print(raw_object.get_data().shape)
    epoch = autoreject_custom.epoch_object_creation(raw_object)
    print(epoch.get_data().shape)
  #print('epochs shape',epoch.get_data().shape)
    clean_epoch,reject_log = autoreject_custom.autoreject_algo(epoch,n_interpolates = np.array([1, 4, 32]),consensus_percs = np.linspace(0, 1.0, 11))
  #clean_epochs.append([clean_epoch,reject_log])
    if clean_epoch.drop_log_stats() == 0:
        clean_epochs[trial+1] = [clean_epoch.get_data(),reject_log.bad_epochs,clean_epoch.drop_log_stats(),dataset.Y[trial],dataset.Z[trial][1]]
  


# In[ ]:


def driver_code():

  #load dataset
    dataset_dict = {0:'DEAP',1:'OASIS',2:'DREAMER'}
    print(dataset_dict)
    print('enter dataset mapping number you want to use')
    mapping = int(input())
    dataset = preprocessing(dataset_dict[mapping])
    dataset.load_arrays()

  #referencing
    print('next step in preprocessing is referencing')
    referencing_dict = {1:'average_referencing'}
    print(referencing_dict)
    print('enter referencing method')
    mapping = int(input())
    if mapping ==1 :
        averaged_data = referencing.average(dataset.X)
    print('next step is applying filters')
    filter_dict = {1:'notch_filter',2:"butter_worth_filter"}
  
  
  #filtering
    applyed_filters = False
    while applyed_filters == False:
        print(filter_dict)
        mapping = int(input())
        print('sampling frequency of dataset is',dataset.freq)
        if mapping == 1 :
            print('enter notch frequency')
            notch_freq = float(input())
            filtered_data = filters.notch_filter(averaged_data,dataset.freq,notch_freq)

        if mapping == 2:
            print('enter lower frequency')
            lfreq = float(input())
            print('enter higher frequency')
            hfreq = float(input())
            filtered_data = filters.butterworth_filter(dataset.X,dataset.freq,lfreq,hfreq)

    print('Do you want to apply filters again?enter y/n')
    boolean = input()
    if boolean == 'n':
        applyed_filters = True
  
    print('do you want to save the data preprocessed so far?y/n')
    boolean = input()
    if boolean == 'y':
        filename = input('enter filename to save as')
        np.savez('preprocessed_data/'+dataset.name.lower()+'/'+filename,X = dataset.X,Y = dataset.Y)

  #if motion artifact correction using gyrscopic data if dataset is oasis
    if dataset.name == 'OASIS':
        print('do you want to use motion artifact removal using gyroscopic data? y/n')
        boolean = input()
        if boolean == 'y':
            no_of_trials = dataset.X.shape[0]
            print('shape of gyroscope data before reshaping is:',dataset.gyroscope.shape)
            (a,b,c) = dataset.gyroscope.shape
            gyroscope_trials = np.reshape (dataset.gyroscope,(a,c,b))# reshaping trials so they are of the shape trials x channels x timepoints

      #iterating over all trials and correcting trial data for motion artifact
            for trial_n in range(no_of_trials):
                print('processing trial no:',trial_n+1)
                trial_data = filtered_data[trial_n]
                gyroscope_trial  = gyroscope_trials[trial_n,:,:] #only acclerometer values extracted for a particular trial
                gyroscope_trial_x = gyroscope_trial[0] # accelerometer x axis reading
                gyroscope_trial_y = gyroscope_trial[1] # accelerometer y axis reading
                gyroscope_trial_z = gyroscope_trial[2] # accelerometer z axis reading
                source_signals,mixing_matrix = source_decomposition.ica(trial_data,dataset.channels,dataset.ch_type,dataset.freq)
                corrected_sources = process_trial(source_signals,gyroscope_trial_x,gyroscope_trial_y,gyroscope_trial_z)

                #corrected sources are projected back into orignal dimensional space of EEG data using mixing matrix
                project_back = np.matmul(mixing_matrix,corrected_sources)
                filtered_data[trial_n] = project_back

        print(filtered_data.shape)
    print('do you want to save the data preprocessed so far?y/n')
    boolean = input()
    if boolean == 'y':
        filename = input('enter filename to save as')
        np.savez('preprocessed_data/'+dataset.name.lower()+'/'+filename,X = dataset.X,Y = dataset.Y)

    if dataset.name == 'OASIS':
        print('do you want to use autoreject? y/n')
        boolean = input()
        if boolean == 'y':
            print('do you want to save this autoreject cleaned data? y/n')
            boolean = input()
            no_of_trials = dataset.X.shape[0]
      '''
      dictionary contains information about each trial
      each trial number i is mapped to a list containing the cleaned epochs given by autoreject,boolena array indicating which epoch was dropped,and 
      a percentage indicating epochs dropped out of total, valence ,arousal rating for  trial and image_id
      '''
      #running autoreject for each trial data

      '''
      autoreject divides each trial data into 5 epochs of 1 sec segment i.e 640 timepoints into 128 timepoints per epochs,and runs algo on each
      epoch,rejecting epochs based on estimated parameters
      '''
        clean_epochs ={}
        for trial in range(no_of_trials):
            print('trial no',trial)
            temp = filtered_data[trial]
            raw_object = autoreject_custom.raw_object_creation(temp,dataset.channels,dataset.ch_type,dataset.freq)
            print(raw_object.get_data().shape)
            epoch = autoreject_custom.epoch_object_creation(raw_object)
            print(epoch.get_data().shape)
            #print('epochs shape',epoch.get_data().shape)
            clean_epoch,reject_log = autoreject_custom.autoreject_algo(epoch,n_interpolates = np.array([1, 4, 32]),consensus_percs = np.linspace(0, 1.0, 11))
            #clean_epochs.append([clean_epoch,reject_log])
            if clean_epoch.drop_log_stats() == 0:
                clean_epochs[trial+1] = [clean_epoch.get_data(),reject_log.bad_epochs,clean_epoch.drop_log_stats(),dataset.Y[trial],dataset.Z[trial][1]]

        if boolean == 'y':
            with open('preprocessed_data/oasis/with_autoreject.p','wb') as file:
            pickle.dump(clean_epochs,file,protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:



def __main__():
    driver_code()


# In[ ]:


__main__()

