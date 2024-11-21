#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
import os


# In[2]:


path = '/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/2nd_test/2nd_test'


# In[3]:


filename = '2004.02.12.10.32.39'
dataset=pd.read_csv(os.path.join(path, filename), sep='\t',header=None)


# In[4]:


dataset


# In[5]:


for i in [0,1,2,3]:
    
    df_bearing=np.array(dataset.iloc[:,i])
    
    plt.figure(figsize=(20, 5))
    plt.plot(df_bearing)

    plt.legend(['Bearing {}'.format(i+1)])

    plt.xlabel("Data-point")
    plt.ylabel("Acceleration")
    plt.title('Bearing {}'.format(i+1))
    plt.show()


# In[6]:


bearing_no=1

bearing_data = np.array(dataset.iloc[:,bearing_no-1])

bearing_data


# In[7]:


temp = bearing_data
temp


# In[8]:


feature_matrix=np.zeros((1,9))
feature_matrix


# In[9]:


def compute_skewness(x):
    
    n = len(x)
    third_moment = np.sum((x - np.mean(x))**3) / n
    s_3 = np.std(x, ddof = 1) ** 3
    return third_moment/s_3


# In[10]:


def compute_kurtosis(x):
    
    n = len(x)
    fourth_moment = np.sum((x - np.mean(x))**4) / n
    s_4 = np.std(x, ddof = 1) ** 4
    return fourth_moment / s_4 - 3


# In[11]:


feature_matrix[0,0] = np.max(temp)
feature_matrix[0,1] = np.min(temp)
feature_matrix[0,2] = np.mean(temp)
feature_matrix[0,3] = np.std(temp, ddof = 1)
feature_matrix[0,4] = np.sqrt(np.mean(temp ** 2))
feature_matrix[0,5] = compute_skewness(temp)
feature_matrix[0,6] = compute_kurtosis(temp)
feature_matrix[0,7] = feature_matrix[0,0]/feature_matrix[0,4]
feature_matrix[0,8] = feature_matrix[0,4]/feature_matrix[0,2]


# In[12]:


feature_matrix


# In[13]:


df = pd.DataFrame(feature_matrix)
df.index=[filename[:-3]]
df


# In[14]:


Time_feature_matrix=pd.DataFrame()

test_set=3

bearing_no=4 # Provide the Bearing number [1,2,3,4] of the Test set

path='/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/3rd_test/4th_test/txt'
for filename in os.listdir(path):
    
    dataset=pd.read_csv(os.path.join(path, filename), sep='\t',header=None)

    bearing_data = np.array(dataset.iloc[:,bearing_no-1])

    feature_matrix=np.zeros((1,9))
    temp = bearing_data
    feature_matrix[0,0] = np.max(temp)
    feature_matrix[0,1] = np.min(temp)
    feature_matrix[0,2] = np.mean(temp)
    feature_matrix[0,3] = np.std(temp, ddof = 1)
    feature_matrix[0,4] = np.sqrt(np.mean(temp ** 2))
    feature_matrix[0,5] = compute_skewness(temp)
    feature_matrix[0,6] = compute_kurtosis(temp)
    feature_matrix[0,7] = feature_matrix[0,0]/feature_matrix[0,4]
    feature_matrix[0,8] = feature_matrix[0,4]/feature_matrix[0,2]
    
    df = pd.DataFrame(feature_matrix)
    df.index=[filename[:-3]]
    
    Time_feature_matrix = pd.concat([Time_feature_matrix,df])


# In[15]:


Time_feature_matrix


# In[16]:


Time_feature_matrix.columns = ['Max','Min','Mean','Std','RMS','Skewness','Kurtosis','Crest Factor','Form Factor']
Time_feature_matrix.index = pd.to_datetime(Time_feature_matrix.index, format='%Y.%m.%d.%H.%M')

Time_feature_matrix = Time_feature_matrix.sort_index()

Time_feature_matrix.to_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_{}_Test_{}.csv'.format(bearing_no,test_set))

Time_feature_matrix


# #### *Bearing dataset visualization*
# 
# * Recording Duration: February 12, 2004 10:32:39 to February 19, 2004 06:22:39
# * No. of Files: 984
# * No. of Channels: 4
# * Channel Arrangement: Bearing 1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing 4 – Ch 4.
# * File Recording Interval: Every 10 minutes
# * File Format: ASCII
# * Description: At the end of the test-to-failure experiment, outer race failure occurred in bearing 1.

# In[17]:


df1 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_1_Test_2.csv")
df1 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_1_Test_2.csv",index_col='Unnamed: 0')
df1.index = pd.to_datetime(df1.index)
df1


# In[18]:


df1 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_1_Test_2.csv",index_col='Unnamed: 0')
df2 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_2_Test_2.csv",index_col='Unnamed: 0')
df3 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_3_Test_2.csv",index_col='Unnamed: 0')
df4 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_4_Test_2.csv",index_col='Unnamed: 0')


# In[19]:


df1.index = pd.to_datetime(df1.index)
df1


# ### *Visualizing data*

# In[20]:


df1.columns


# In[21]:


for col in (df1.columns):  
    
        plt.figure(figsize=(8, 5))
        plt.plot(df1.index,df1[col])
        plt.plot(df1.index,df2[col])
        plt.plot(df1.index,df3[col])
        plt.plot(df1.index,df4[col])

        plt.legend(['bearing-1','bearing-2','bearing-3','bearing-4'])

        plt.xlabel("Date-Time")
        plt.ylabel(col)
        plt.title(col)
        plt.show()


# In[ ]:




