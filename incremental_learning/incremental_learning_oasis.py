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
from sklearn.svm import SVR 
from sklearn.metrics import accuracy_score


# In[ ]:


# Either one of the MinMaxScaling or StandardScaling function can be used

scaler_min_max = MinMaxScaler()
scaler_standard = StandardScaler()
def MinMaxScaling(feature_matrix):
    global scaler_min_max
    scaler_min_max.fit(feature_matrix)
    return scaler_min_max.transform(feature_matrix)

def StandardScaling(feature_matrix):
    global scaler_standard
    scaler_standard.fit(feature_matrix)
    print('scaling shape',scaler_standard.mean_.shape)
    return scaler.transform(feature_matrix)


# """##OASIS dataset
# 1> Valence - features selected
# >
# *   HjorthMob
# *    HjorthComp
# *   stdDev
# 
# 2> Arousal - feature selected
# >
# *   HjorthMob
# """

# In[ ]:


# now for incremental learning we need to segregate data of subjects
def segregate_data_of_subjects(feature_matrix,Y,total_subjects,sfreq = 128):
  '''
  returns a dictionary which contains the samples data only corresponding to particular subjects of feature matrix
  '''
# parameters :-
            # feature_matrix :- Vector containing the features mentioned above subject wise, to be used for cross validation
            # Y :- The Valence and Arousal values as entered by the subjects
            # total_subjects :- Total number of subjects in the study
            # sfreq :- sampling frequency of the EEG data
# returns :-
            # subject_indexes :- Subject wise features in a dictionary form
            # aligned_y :- the y values corresponding to each subject
        
    subject_indexes = { 'subject_1':feature_matrix[:200],
                      'subject_2':feature_matrix[200:400],
                      'subject_3':feature_matrix[400:600],
                      'subject_4':feature_matrix[600:795],
                      'subject_5':feature_matrix[795:995],
                      'subject_6':feature_matrix[995:1185],
                      'subject_7':feature_matrix[1185:1375],
                      'subject_8':feature_matrix[1375:1575],
                      'subject_9':feature_matrix[1575:1770],
                      'subject_10':feature_matrix[1770:1965],
                      'subject_11':feature_matrix[1965:2160],
                      'subject_12':feature_matrix[2160:2360],
                      'subject_13':feature_matrix[2360:2550],
                      'subject_14':feature_matrix[2550:2740],
                      'subject_15':feature_matrix[2740:2935]
                      }

    aligned_y =   { 'subject_1':Y[:200],
                      'subject_2':Y[200:400],
                      'subject_3':Y[400:600],
                      'subject_4':Y[600:795],
                      'subject_5':Y[795:995],
                      'subject_6':Y[995:1185],
                      'subject_7':Y[1185:1375],
                      'subject_8':Y[1375:1575],
                      'subject_9':Y[1575:1770],
                      'subject_10':Y[1770:1965],
                      'subject_11':Y[1965:2160],
                      'subject_12':Y[2160:2360],
                      'subject_13':Y[2360:2550],
                      'subject_14':Y[2550:2740],
                      'subject_15':Y[2740:2935]
                      }


    return subject_indexes,aligned_y


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

  #now we also need to extract the valence/arousal data for the corresponding subject
    y = np.empty((0))
    for subject in range(number_of_subjects):
        subject_name = 'subject_'+str(subject+1)
        subject_y_data = Y[subject_name]
        y = np.append(y,subject_y_data,axis=0)

  
    print('shape of X is :',X.shape)
    print('shape of y is :',y.shape)
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

  #extracting feature data related to valence
    HjorthMob_v = np.load('features/oasis/with_autoreject/Hjorth_mobilty_0_0.npz')
    HjorthComp_v = np.load('features/oasis/with_autoreject/Hjorth_complexity_0_0.npz')
    stdDev_v = np.load('features/oasis/with_autoreject/stdDev_0_0.npz')
  
  #creating feature matrix 
    feature_matrix_valence = np.empty((0,2935))
    feature_matrix_valence = np.append(feature_matrix_valence,HjorthMob_v['features'],axis = 0)
    feature_matrix_valence = np.append(feature_matrix_valence,HjorthComp_v['features'],axis = 0)
    feature_matrix_valence = np.append(feature_matrix_valence,stdDev_v['features'],axis = 0)
    feature_matrix_valence = feature_matrix_valence.T #shape of feature matrix is 2935 x 42

  #extracting valence labels
    Y_val = HjorthMob_v['Y'][:,0]

  #extracting feature data related to arousal
    HjorthMob_a = np.load('features/oasis/with_autoreject/Hjorth_mobilty_0_0.npz')

  #creating feature matrix for arousal
    feature_matrix_arousal = np.empty((0,2935))
    feature_matrix_arousal = np.append(feature_matrix_arousal,HjorthMob_a['features'],axis=0)
    feature_matrix_arousal = feature_matrix_arousal.T

  #extracting arousal labels
    Y_aro = HjorthMob_a['Y'][:,1]

    model = SVR() #initialize model

  #running incremental learning loop for valence
    print('')
    print('Incremental training for valence')
    print('')
    test_subject = []
    rmse_val = []
    subject_indexes_valence,aligned_Y_val = segregate_data_of_subjects(feature_matrix_valence,Y_val,15,128)
    i = 1
    while i <= 15:
        rmse_val,test_subject= training_phase(model,feature_matrix_valence,aligned_Y_val,subject_indexes_valence,i,15,rmse_val,test_subject)
        i+=1

  #running incremental learning loop for arousal
    print('')
    print('Incremental training for arousal ')
    print(' ')

    model = SVR()#reinitialize model
    test_subject = []
    rmse_aro = []
    subject_indexes_arousal,aligned_Y_aro = segregate_data_of_subjects(feature_matrix_arousal,Y_aro,15,128)
    i=1
    while i<=15:
        rmse_aro,test_subject = training_phase(model,feature_matrix_arousal,aligned_Y_aro,subject_indexes_arousal,i,15,rmse_aro,test_subject)
        i+=1

  
    fig,axe = plt.subplots(1,1,figsize = (40,20))
    axe.plot(test_subject,rmse_val,color='r',label = 'rmse valence')
    axe.plot(test_subject,rmse_aro,color = 'g',label = 'rmse arousal')
    axe.set_xlabel('trained upto subject')
    axe.set_ylabel('rmse')
    axe.set_title('support vector regressor')
    axe.legend(loc = 'upper right')

    df = pd.DataFrame([rmse_val,rmse_aro],columns = test_subject,index = ['valence rms','arousal rms'])
  
    if save == 'y':
        plt.savefig('plots/oasis/all_feature_valence_arousal_rmse',format="svg")
        df.to_csv('plots/oasis/all_features_valence_arousal_rmse.csv')


# In[ ]:


if __name__ == '__main__':
    driver_code(sys.argv[1])

