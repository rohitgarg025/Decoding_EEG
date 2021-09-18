#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[ ]:


scaler_min_max = MinMaxScaler()
scaler_standard = StandardScaler()

# Either one of the MinMaxScaling or StandardScaling function can be used

def MinMaxScaling(feature_matrix):
    global scaler_min_max
    scaler_min_max.fit(feature_matrix)
    return scaler_min_max.transform(feature_matrix)

def StandardScaling(feature_matrix):
    global scaler_standard
    scaler_standard.fit(feature_matrix)
    print('scaling shape',scaler_standard.mean_.shape)
    return scaler.transform(feature_matrix)

architecture = 'sklearn'
if architecture == 'sklearn':
    from sklearn.svm import SVR 
    from sklearn.metrics import accuracy_score
else: 
    from cuml.svm import SVR
    from cuml.metrics import  accuracy_score


# 
# """##DEAP dataset
# 1> Valence - features selected
# >
# *   bandPwr_gamma
# *   bandPwr_beta
# *   ShannonRes_gamma
# *   ShannonRes_beta
# *   rasm_gamma
# *   dasm_gamma
# 
# 2> Arousal - feature selected
# >
# *   HjorthMob
# *   HjorthComp
# *   stdDev
# *   bandPwr_theta
# *   bandPwr_beta
# *   ShannonRes_beta
# *   ShannonRes_gamma
# *   dasm_beta
# """

# In[ ]:


# now for incremental learning we need to segregate data of subjects
def segregate_data_of_subjects(feature_matrix,total_subjects,sfreq = 128):
  '''
  reuturs a dictionary which contains the samples data only corresponding to particular subjects of feature matrix
  '''
# parameters :-
            # feature_matrix :- Vector containing the features mentioned above subject wise, to be used for cross validation
            # total_subjects :- Total number of subjects in the study
            # sfreq :- sampling frequency of the EEG data
# returns :-
            # subject_indexes :- Subject wise features in a dictionary form
    
    total_samples = feature_matrix.shape[0]
    subject_indexes = {}
    samples_per_subject = total_samples//total_subjects
    for i in range(total_subjects):
        subject_name = 'subject_' + str(i+1)
        subject_indexes[subject_name] = feature_matrix[samples_per_subject*i:samples_per_subject*(i+1),:]
  
    return subject_indexes


# In[ ]:


# now defining a function which carries out the incremenatal learning algo
def training_phase(model,feature_matrix,Y,subject_indexes,number_of_subjects,total_subjects,rmse_score,test_subject):
# parameters :-
            # model :- The training model to be used (SVR in this case)
            # featrue_matrix :- feature matrix obtained in the above function
            # Y :- The Valence and Arousal values as entered by the subjects
            # subject_indexes :-Subject wise features in a dictionary form
            # number_of_subjects :- Total number of subjects in the study
            # total_subjects :- Total number of subjects in the study
            # rmse_score :- RMSE of the previous iterations 
            # test_subject :- Cross validation test subject list

# returns :-
            # rmse_score :- Array of rmse scores over the iterations, updated with the rmse score of the current iteration
            # test_subject :- Updated Cross validation test subject list
    no_of_features = feature_matrix.shape[1]
    X = np.empty((0,no_of_features))
    print('training on subject_no:',end = ' ')

  #create a feature matrix containing data upto subjects given by the number number_of_subjects
  #for eg if number of subject ==4 , data of first 4 subjects will be taken and a feature matrix made out of it to feed to the ml model
  
    for subject in range(number_of_subjects):
        print(subject+1,end = ' ')
        subject_name = 'subject_'+str(subject+1)
        subject_data = subject_indexes[subject_name]
        X = np.append(X,subject_data,axis=0)
    print(' ')

  #apply a MinMax scaling to the current iteration feature matrix
    X = MinMaxScaling(X)

  #now we also need to extract the valence arousal data for the corresponding subject
    y = np.empty((0))
    total_samples = feature_matrix.shape[0]
    samples_per_subject = total_samples//total_subjects
    for subject in range(number_of_subjects):
        y = Y[:samples_per_subject*(number_of_subjects)]

    print('shape of X is :',X.shape)
    print('shape of y is  :',y.shape)

  #shuffling data randomly to feed to model
    X,y = shuffle(X,y,random_state = 0)

  #doing a train test split of 80:20
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)

  #training_model
    model = model.fit(X_train,y_train)

  #testing_model
    y_predict = model.predict(X_test)


  #calculating rmse values for valence and arousal using model fitted for current iteration
    y_rms = np.sqrt(mean_squared_error(y_test,y_predict))
    print('rms on y :',y_rms)
    print('')
    rmse_score.append(y_rms)
    test_subject.append(subject_name)

    return rmse_score,test_subject


# In[ ]:


def driver_code(save):
    
    # Function to load the features, then train the regressor and will give the validation and test plot

  #extracting file data corresponding to valence features
    bandPwr_gamma_v = np.load('features/deap/bandPwr_gamma_1_1.npz')
    bandPwr_beta_v = np.load('features/deap/bandPwr_beta_1_1.npz')
    ShannonRes_gamma_v = np.load('features/deap/ShannonRes_sub_bands_gamma_1_1.npz')
    ShannonRes_beta_v = np.load('features/deap/ShanninRes_sub_bands_beta_1_1.npz')
    rasm_gamma_v = np.load('features/deap_RASM_DASM/gamma_1_1.npz')#shape of feature is 80640 x 14, be careful to extract only rasm features, i.e first 7 columns
    dasm_gamma_v = np.load('features/deap_RASM_DASM/gamma_1_1.npz')

  #creating a feature matrix for valence
    feature_matrix_valence = np.empty((0,80640))
    feature_matrix_valence = np.append(feature_matrix_valence,bandPwr_gamma_v['features'],axis = 0)
    feature_matrix_valence = np.append(feature_matrix_valence,bandPwr_beta_v['features'],axis = 0)
    feature_matrix_valence = np.append(feature_matrix_valence,ShannonRes_gamma_v['features'],axis = 0)
    feature_matrix_valence = np.append(feature_matrix_valence,ShannonRes_beta_v['features'],axis = 0)
    feature_matrix_valence = np.append(feature_matrix_valence,rasm_gamma_v['features'].T[:7,:],axis = 0)
    feature_matrix_valence = np.append(feature_matrix_valence,dasm_gamma_v['features'].T[7:,:],axis = 0)
    feature_matrix_valence = feature_matrix_valence.T#feature matrix is of shape 80640 x 70
  
  #extracting labels 
    Y_val = bandPwr_gamma_v['Y'][:,0]

  #extracting file data corresponding to arousal features
    HjorthMob_a = np.load('features/deap/Hjorth_mobilty_1_1.npz')
    HjorthComp_a = np.load('features/deap/Hjorth_complexity_1_1.npz')
    stdDev_a = np.load('features/deap/stdDev_1_1.npz')
    bandPwr_beta_a = np.load('features/deap/bandPwr_beta_1_1.npz')
    bandPwr_theta_a = np.load('features/deap/bandPwr_theta_1_1.npz')
    ShannonRes_beta_a = np.load('features/deap/ShanninRes_sub_bands_beta_1_1.npz')
    ShannonRes_gamma_a = np.load('features/deap/ShannonRes_sub_bands_gamma_1_1.npz')
    dasm_beta_a = np.load('features/deap_RASM_DASM/beta_1_1.npz')

  #creating feature matrix for arousal
    feature_matrix_arousal = np.empty((0,80640))
    feature_matrix_arousal = np.append(feature_matrix_arousal,HjorthMob_a['features'],axis = 0)
    feature_matrix_arousal = np.append(feature_matrix_arousal,HjorthComp_a['features'],axis = 0)
    feature_matrix_arousal = np.append(feature_matrix_arousal,stdDev_a['features'],axis = 0)
    feature_matrix_arousal = np.append(feature_matrix_arousal,bandPwr_beta_a['features'],axis = 0)
    feature_matrix_arousal = np.append(feature_matrix_arousal,bandPwr_theta_a['features'],axis = 0)
    feature_matrix_arousal = np.append(feature_matrix_arousal,ShannonRes_beta_a['features'],axis = 0)
    feature_matrix_arousal = np.append(feature_matrix_arousal,ShannonRes_gamma_a['features'],axis = 0)
    feature_matrix_arousal = np.append(feature_matrix_arousal,dasm_beta_a['features'].T[7:,:],axis = 0)
    feature_matrix_arousal = feature_matrix_arousal.T#shape of feature matrix is 80640 x 105
  
  #extracting labels
    Y_aro = HjorthMob_a['Y'][:,1]

    model = SVR()#initializing support vector regressor for training

  #running incremental learning loop for valence
    print('')
    print('Incremental training for valence')
    print('')
    test_subject = []
    rmse_val = []
    subject_indexes_valence = segregate_data_of_subjects(feature_matrix_valence,32,128)
    i = 1
    while i <= 32:
        rmse_val,test_subject= training_phase(model,feature_matrix_valence,Y_val,subject_indexes_valence,i,32,rmse_val,test_subject)
        i+=1

  #running incremental learning loop for arousal
    print('')
    print('Incremental training for arousal ')
    print(' ')

    model = SVR()#reinitialize model
    test_subject = []
    rmse_aro = []
    subject_indexes_arousal = segregate_data_of_subjects(feature_matrix_arousal,32,128)
    i=1
    while i<=32:
        rmse_aro,test_subject = training_phase(model,feature_matrix_arousal,Y_aro,subject_indexes_arousal,i,32,rmse_aro,test_subject)
        i+=1

  
    fig,axe = plt.subplots(1,1,figsize = (40,20))
    axe.plot(test_subject,rmse_val,color='r',label='rmse valence')
    axe.plot(test_subject,rmse_aro,color = 'g',label='rmse arousal')
    axe.set_xlabel('trained upto subject')
    axe.set_ylabel('rmse')
    axe.set_title('support vector regressor')
    axe.legend(loc = 'upper right')

    df = pd.DataFrame([rmse_val,rmse_aro],columns = test_subject,index = ['valence rms','arousal rms'])
    print(df)
  
    if save == 'y':
        plt.savefig('plots/deap/all_feature_valence_arousal_rmse',format = "svg")
        df.to_csv('plots/deap/all_features_valence_arousal_rmse.csv')


# In[ ]:


if __name__ == '__main__':
    driver_code(sys.argv[1])

