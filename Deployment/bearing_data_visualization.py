#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
import os


# #### Test-1
# * Channel Arrangement: Bearing 1 – Ch 1&2; Bearing 2 – Ch 3&4;Bearing 3 – Ch 5&6; Bearing 4 – Ch 7&8.
# * At the end of the test-to-failure experiment, inner race defect occurred in bearing 3 and roller element defect in bearing 4.

# In[2]:


test_no=1

df1 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_1_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df2 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_2_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df3 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_3_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df4 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_4_Test_{}.csv".format(test_no),index_col='Unnamed: 0')

df1.index = pd.to_datetime(df1.index)


# In[3]:


for i,col in enumerate(df1.columns):  
    
        plt.figure(figsize=(15, 5))
        plt.plot(df1.index,df1[col])
        plt.plot(df1.index,df2[col])
        plt.plot(df1.index,df3[col])
        plt.plot(df1.index,df4[col])

        plt.legend(['bearing-1','bearing-2','bearing-3','bearing-4'])

        plt.xlabel("Date-Time")
        plt.ylabel(col)
        plt.title(col)
        plt.show()


# ### *Save inner race fault*

# In[4]:


df_irf=df3['2003-11-21 00:32:00':'2003-11-24 18:22:00']
fault=[]
for i in range (0,len(df_irf)):
    fault.append('Inner Race')

df_irf['Fault']=fault
df_irf


# In[5]:


df_irf.to_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/inner_race_fault.csv',index=False)


# #### *Saving roller element fault*

# In[6]:


df_ref=df4['2003-11-21 00:32:00':'2003-11-24 18:22:00']
fault=[]
for i in range (0,len(df_ref)):
    fault.append('Roller Element')

df_ref['Fault']=fault
df_ref


# In[7]:


df_ref.to_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/roller_element_fault.csv',index=False)


# ### Test 2   MORE ACCURATE
# * Channel Arrangement: Bearing 1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing 4 – Ch 4.
# * At the end of the test-to-failure experiment, outer race failure occurred in bearing 1.

# In[8]:


test_no=2

df1 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_1_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df2 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_2_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df3 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_3_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df4 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_4_Test_{}.csv".format(test_no),index_col='Unnamed: 0')

df1.index = pd.to_datetime(df1.index)

for i,col in enumerate(df1.columns):  
    
        plt.figure(figsize=(15, 5))
        plt.plot(df1.index,df1[col])
        plt.plot(df1.index,df2[col])
        plt.plot(df1.index,df3[col])
        plt.plot(df1.index,df4[col])

        plt.legend(['bearing-1','bearing-2','bearing-3','bearing-4'])

        plt.xlabel("Date-Time")
        plt.ylabel(col)
        plt.title(col)
        plt.show()


# ### *Saving outer race fault*

# In[9]:


df_orf=df1['2004-02-17 12:32:00':'2004-02-19 00:42:00']
fault=[]
for i in range (0,len(df_orf)):
    fault.append('Outer Race')

df_orf['Fault']=fault

df_orf.to_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_2.csv',index=False)  
df_orf


# ### Test 3
# * Bearing1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing4 – Ch4
# * At the end of the test-to-failure experiment, outer race failure occurred in bearing 3.

# In[10]:


test_no=3

df1 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_1_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df2 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_2_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df3 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_3_Test_{}.csv".format(test_no),index_col='Unnamed: 0')
df4 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_4_Test_{}.csv".format(test_no),index_col='Unnamed: 0')

df1.index = pd.to_datetime(df1.index)

for i,col in enumerate(df1.columns):  
    
        plt.figure(figsize=(10, 5))
        plt.plot(df1.index,df1[col])
        plt.plot(df1.index,df2[col])
        plt.plot(df1.index,df3[col])
        plt.plot(df1.index,df4[col])

        plt.legend(['bearing-1','bearing-2','bearing-3','bearing-4'])

        plt.xlabel("Date-Time")
        plt.ylabel(col)
        plt.title(col)
        plt.show()


# ### Saving outer race fault

# In[11]:


df3


# In[12]:


df_orf=df3['2004-04-15 12:32:00':'2004-04-18 00:42:00']
fault=[]
for i in range (0,len(df_orf)):
    fault.append('Outer Race')

df_orf['Fault']=fault

df_orf.to_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_3.csv',index=False)  
df_orf


# ## Collecting Normal Data from all the Bearings

# In[13]:


Test=[1,2,3]
Bearing_No=[1,2,3,4]

df_normal_bearing = pd.DataFrame()

for test_no in Test:
    for bearing_no in Bearing_No:
        temp = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_{}_Test_{}.csv".format(bearing_no,test_no),index_col='Unnamed: 0')

        starting = int(np.floor(len(temp)*(21/100)))
        ending = int(np.floor(len(temp)*(23/100)))

        start_time = temp.index[starting]
        end_time = temp.index[ending]

        temp = temp[start_time:end_time]
        
        df_normal_bearing= pd.concat([df_normal_bearing,temp])

fault=[]
for i in range (0,len(df_normal_bearing)):
    fault.append('Normal')

df_normal_bearing['Fault']=fault

df_normal_bearing.to_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Normal_Bearing.csv',index=False) 


# In[14]:


df_normal_bearing


# In[ ]:




