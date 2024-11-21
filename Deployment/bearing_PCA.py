#!/usr/bin/env python
# coding: utf-8

# ### *Dimensionality Reduction* (Because we have 9 features and for optimal results we have to reduce these features to approx 2 or 3 features)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use
import matplotlib
matplotlib.use('TkAgg')  # or 'MacOSX', 'agg'



# In[2]:


df1=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Normal_Bearing.csv')
df2=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_2.csv')
df3=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_3.csv')
df4=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/inner_race_fault.csv')
df5=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/roller_element_fault.csv')

df = pd.concat([df1,df2,df3,df4,df5])
df = df.reset_index(drop=True)
df


# In[3]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# ## *PCA with 2 components*

# In[4]:


from sklearn.preprocessing import StandardScaler #(pca requires high scale therefore standardscalar is used)
X = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca_2 = PCA(n_components=2)
X_pca = pca_2.fit_transform(X)

principalDf = pd.DataFrame(data = X_pca, columns = ['principal_component_1', 'principal_component_2'])

principalDf['Fault']=np.array(df['Fault'])


# In[5]:


principalDf


# In[6]:


principalDf['Fault'].unique()


# In[7]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal_Component_1', fontsize = 15)
ax.set_ylabel('Principal_Component_2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
fault = ['Normal', 'Outer Race', 'Inner Race', 'Roller Element']
colors = ['g', 'r','b','y']
for fault, color in zip(fault,colors):
    indicesToKeep = principalDf['Fault'] == fault
    ax.scatter(principalDf.loc[indicesToKeep, 'principal_component_1']
               , principalDf.loc[indicesToKeep, 'principal_component_2']
               , c = color
               , s = 50)
ax.legend(['Normal', 'Outer Race', 'Inner Race', 'Roller Element'])


# In[8]:


sum(pca_2.explained_variance_ratio_)
pca_2.explained_variance_ratio_


# ### *PCA with 3 components* 

# *since pca with 2 is too complicated and sum variance is around 75 which is less therefore we will perform pca with 3 to get maximum variance*

# In[9]:


from mpl_toolkits.mplot3d import Axes3D


# In[10]:


X = StandardScaler().fit_transform(X)

pca_3 = PCA(n_components=3)

X_pca = pca_3.fit_transform(X)

principalDf = pd.DataFrame(data = X_pca, columns = ['principal_component_1', 'principal_component_2','principal_component_3'])

principalDf['Fault']=np.array(df['Fault'])


# In[11]:


principalDf


# In[16]:

use('Qt5Agg')


fig = plt.figure(figsize=(10,14))
  
# syntax for 3-D projection

ax = plt.axes(projection='3d')
  
# defining all 3 axes
fault = ['Normal', 'Outer Race', 'Inner Race', 'Roller Element']
colors = ['g', 'r','b','y']
for fault, color in zip(fault,colors):
    indicesToKeep = principalDf['Fault'] == fault
    ax.scatter3D(principalDf.loc[indicesToKeep, 'principal_component_1']
               , principalDf.loc[indicesToKeep, 'principal_component_2']
               , principalDf.loc[indicesToKeep, 'principal_component_3']
               , c = color
               , s = 50)
ax.legend(['Normal', 'Outer Race', 'Inner Race', 'Roller Element'])

  
# plotting
ax.set_xlabel('Principal_Component_1', fontsize = 15)
ax.set_ylabel('Principal_Component_2', fontsize = 15)
ax.set_zlabel('Principal_Component_3', fontsize = 15)

ax.set_title('3D PCA')
ax.view_init(45,90)

plt.show()


# In[13]:


np.sum(pca_3.explained_variance_ratio_)
pca_3.explained_variance_ratio_


# *now since pca with 3 is turned out to be 85 which is better than pca 2 which was 75 therefore we will concider pca 3*

# In[14]:


X = StandardScaler().fit_transform(X)

pca_4 = PCA(n_components=4)

X_pca = pca_4.fit_transform(X)

principalDf = pd.DataFrame(data = X_pca, columns = ['principal_component_1', 'principal_component_2','principal_component_3' , 'principal_component_4'])

principalDf['Fault']=np.array(df['Fault'])


# In[15]:


#np.sum(pca_4.explained_variance_ratio_)
#pca_4.explained_variance_ratio_ #TRIED PCA WITH 4 COMPONENTS


# In[ ]:




