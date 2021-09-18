#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/gdrive',force_remount = True)


# In[ ]:


get_ipython().run_line_magic('cd', '../gdrive/MyDrive/emotion_recognition_project/')


# # Incremental Learning for OASIS

# **Arguments**
# 
# 
# ---
# 
# save = 'y/n'
# 
# ---
# 
# 
# 
# 
# 
# Eg. if you want to run model on OASIS dataset,don't want to save plots, with 
# command would be 
# 
# !python incremental_learning_OASIS.py n
# 
# 
# 
# 
# 

# In[ ]:


get_ipython().system('python incremental_learning_oasis.py n')


# 
# # Incremental Learning for DEAP

# **Arguments**
# 
# 
# ---
# 
# save = 'y/n'
# 
# ---
# 
# 
# 
# 
# 
# Eg. if you want to run model on DEAP dataset,don't want to save plots, with 
# command would be 
# 
# !python incremental_learning_DEAP.py n

# In[ ]:


get_ipython().system('python incremental_learning_deap.py n ')


# 
# # Incremental Learning for DREAMER

# **Arguments**
# 
# 
# ---
# 
# save = 'y/n'
# 
# ---
# 
# 
# 
# 
# 
# Eg. if you want to run model on DEAP dataset,don't want to save plots, with 
# command would be 
# 
# !python incremental_learning_DREAMER.py n

# In[ ]:


get_ipython().system('python incremental_learning_dreamer.py n ')


# In[ ]:




