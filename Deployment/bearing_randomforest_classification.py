#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


df1=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Normal_Bearing.csv')
df2=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_2.csv')
df3=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_3.csv')
df4=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/inner_race_fault.csv')
df5=pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/roller_element_fault.csv')


# In[3]:


df = pd.concat([df1,df2,df3,df4,df5])
df


# In[4]:


#sns.set(rc={'figure.figsize':(10,6)})
#sns.scatterplot(x='Form Factor',y='Max',hue='Fault',data=df,palette='Dark2_r')


# In[5]:


df['Fault'].unique()


# In[6]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20)


# In[8]:


#using random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
  
labels=['Normal', 'Outer Race', 'Inner Race', 'Roller Element']    
    
# creating a confusion matrix
cm = confusion_matrix(y_test, y_pred,labels=['Normal', 'Outer Race', 'Inner Race', 'Roller Element'], normalize ='true')
colormap = sns.color_palette("Reds")
sns.heatmap(cm, annot=True,cmap=colormap,xticklabels=labels, yticklabels=labels)


# ## *Testing the Random Forest Model*
# * At the end of the TEST-1, inner race defect occurred in bearing 3 and roller element defect in bearing 4.
# * At the end of the TEST-2, outer race failure occurred in bearing 1.
# * At the end of the TEST-3, outer race failure occurred in bearing 3.

# In[9]:


Test_no=2
Bearing_no=1

test_2 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_{}_Test_{}.csv".format(Bearing_no,Test_no),index_col='Unnamed: 0')


# In[10]:


y_pred_test_2 = rf_model.predict(test_2)
y_pred_test_2


# In[11]:


test_2['Fault']=y_pred_test_2


# In[12]:


test_2


# In[13]:


test_2.index = pd.to_datetime(test_2.index)


# In[14]:


Test_no=2
Bearing_no=1


test_2 = pd.read_csv("/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_{}_Test_{}.csv".format(Bearing_no,Test_no),index_col='Unnamed: 0')
y_pred_test_2 = rf_model.predict(test_2)
test_2['Fault']=y_pred_test_2
test_2.index = pd.to_datetime(test_2.index)


norm = test_2[test_2['Fault']=='Normal']
Out_rac = test_2[test_2['Fault']=='Outer Race']
iner_rac = test_2[test_2['Fault']=='Inner Race']
roll_elem = test_2[test_2['Fault']=='Roller Element']

###############################################################

col='Max'                      # can change this valur to mis ,std ,etc.
plt.figure(figsize=(10, 5))
plt.scatter(norm.index,norm[col])
plt.scatter(Out_rac.index,Out_rac[col])
plt.scatter(iner_rac.index,iner_rac[col])
plt.scatter(roll_elem.index,roll_elem[col])

plt.legend(['Normal','Outer Race','Inner Race','Roller Element'])
plt.title(col)
plt.show()


# #### *Instead of writing above code block , we can use the below goven code from seaborn library to visualising data*

# In[15]:


#sns.set(rc={'figure.figsize':(10,6)})
#sns.scatterplot(x=test_2.index,y='Max',hue='Fault',data=test_2,palette='Dark2_r')


# In[ ]:




