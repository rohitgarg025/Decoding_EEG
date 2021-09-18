#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


from google.colab import drive
drive.mount('/gdrive',force_remount=True)


# In[ ]:


get_ipython().run_line_magic('cd', '/gdrive/MyDrive/emotion_recognition_project/')


# Script to obtain the incremental learning graph for the DEAP, DREAMER and OASIS datasets.

# ##plots for DEAP

# In[ ]:


dataset_deap=glob.glob('plots/deap/*.csv')


# In[ ]:


dataset_deap


# In[ ]:


dataset_svr_deap = pd.read_csv(dataset_deap[0]).T
dataset_svr_deap.columns = ['valence','arousal']
dataset_svr_deap = dataset_svr_deap.drop('Unnamed: 0')
dataset_svr_deap= dataset_svr_deap[::1]
x_deap = range(1,33,1)
dataset_svr_deap


# In[ ]:


fig_deap,axe_deap = plt.subplots(1,1,figsize = (17,10))
axe_deap.plot(x_deap,dataset_svr_deap['valence'],color='green',marker = 'x',markersize=10)
axe_deap.plot(x_deap,dataset_svr_deap['arousal'],color ='red',marker = 'x',markersize=10)
axe_deap.legend(['rfr_valence','rfr_arousal'],)
axe_deap.set_xlabel('trained upto subject')
axe_deap.set_ylabel('RMSE values')
plt.rcParams.update({'font.size':40})
plt.tight_layout()
plt.xticks(x_deap[::3])


# In[ ]:


fig_deap.savefig('final_plots/deap_rfr__valence_arousal_rms.svg')
fig_deap.savefig('final_plots/deap_rfr__valence_arousal_rms.png')


# ##plots for DREAMER

# In[ ]:


dataset_dreamer=glob.glob('plots/dreamer/*.csv')


# In[ ]:


dataset_dreamer


# In[ ]:


dataset_svr_dreamer = pd.read_csv(dataset_dreamer[0]).T
dataset_svr_dreamer.columns = ['valence','arousal']
dataset_svr_dreamer = dataset_svr_dreamer.drop('Unnamed: 0')
x_dreamer = range(1,24,1)
dataset_svr_dreamer= dataset_svr_dreamer[::1]
dataset_svr_dreamer


# In[ ]:


fig_dreamer,axe_dreamer = plt.subplots(1,1,figsize=(17,10))
axe_dreamer.plot(x_dreamer,dataset_svr_dreamer['valence'],color='green',marker = 'x',markersize=10)
axe_dreamer.plot(x_dreamer,dataset_svr_dreamer['arousal'],color ='red',marker = 'x',markersize=10)
axe_dreamer.legend(['rfr_valence','rfr_arousal'],)
axe_dreamer.set_xlabel('trained upto subject')
axe_dreamer.set_ylabel('RMSE values')
plt.rcParams.update({'font.size':40})
plt.tight_layout()
plt.xticks(x_dreamer[::3])


# In[ ]:


fig_dreamer.savefig('final_plots/dreamer_rfr__valence_arousal_rms.svg')
fig_dreamer.savefig('final_plots/dreamer_rfr__valence_arousal_rms.png')


# ##plots for oasis

# In[ ]:


dataset_oasis=glob.glob('plots/oasis/*.csv')


# In[ ]:


dataset_oasis


# In[ ]:


dataset_svr_oasis = pd.read_csv(dataset_oasis[0]).T
dataset_svr_oasis.columns = ['valence','arousal']
dataset_svr_oasis = dataset_svr_oasis.drop('Unnamed: 0')
x_oasis = range(1,16,1)
dataset_svr_oasis= dataset_svr_oasis[::1]
dataset_svr_oasis


# In[ ]:


fig_oasis,axe_oasis = plt.subplots(1,1,figsize=(17,10))
axe_oasis.plot(x_oasis,dataset_svr_oasis['valence'],color='green',marker = 'x',markersize=10)
axe_oasis.plot(x_oasis,dataset_svr_oasis['arousal'],color ='red',marker = 'x',markersize=10)
axe_oasis.set_xlabel('trained upto subject')
axe_oasis.set_ylabel('RMSE values')
axe_oasis.legend(['rfr_valence','rfr_arousal'],loc = 'lower right')
plt.rcParams.update({'font.size':40})
plt.xticks(x_oasis[::3])
plt.tight_layout()


# In[ ]:


fig_oasis.savefig('final_plots/oasis_rfr__valence_arousal_rms.svg')
fig_oasis.savefig('final_plots/oasis_rfr__valence_arousal_rms.png')


# In[ ]:





# In[ ]:


f,a = plt.subplots(3,1,figsize = (40,30))
a[0].plot(x_deap,dataset_svr_deap['valence'],color='green',marker = 'x',markersize=10)
a[0].plot(x_deap,dataset_svr_deap['arousal'],color ='red',marker = 'x',markersize=10)
a[0].legend(['svr_valence','svr_arousal','rfr_valence','rfr_arousal'],)
#a[0].set_xlabel('trained upto subject')
a[0].set_ylabel('RMSE values')
a[0].set_title('DEAP')
a[1].plot(x_dreamer,dataset_svr_dreamer['valence'],color='green',marker = 'x',markersize=10)
a[1].plot(x_dreamer,dataset_svr_dreamer['arousal'],color ='red',marker = 'x',markersize=10)
#a[1].legend(['svr_valence','svr_arousal','rfr_valence','rfr_arousal'],)
#a[1].set_xlabel('trained upto subject')
a[1].set_ylabel('RMSE values')
a[1].set_title('DREAMER')
a[2].plot(x_oasis,dataset_svr_oasis['valence'],color='green',marker = 'x',markersize=10)
a[2].plot(x_oasis,dataset_svr_oasis['arousal'],color ='red',marker = 'x',markersize=10)
a[2].set_xlabel('trained upto subject')
a[2].set_ylabel('RMSE values')
#a[2].legend(['svr_valence','svr_arousal','rfr_valence','rfr_arousal'],loc = 'lower right')
a[2].set_title('OASIS')
plt.rcParams.update({'font.size':40})
plt.tight_layout()


# In[ ]:


f.savefig('final_plots/all_plots_incremental learning.svg')


# In[ ]:




