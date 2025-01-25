#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[5]:


#printing the information
data.info()


# In[6]:


#Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)

data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1
# In[7]:


data1 = data.drop(['Unnamed: 0','Temp C'], axis =1)
data1


# In[8]:


data['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[9]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[10]:


data1.drop_duplicates(keep='first',inplace = True)
data1


# In[12]:


#RENAMING THE COLUMNS
data1.rename({'Solar.R': 'Solar'},axis=1,inplace = True)
data1


# In[ ]:


#Impute the missing value in the table


# In[13]:


data1.info()


# In[14]:


#Display data1 missing values count in each colimn using isnull().sum()
data1.isnull().sum()


# In[16]:


#visualize data1 missing value using heat map
cols = data1.columns
colors = ['black','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[17]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[18]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[20]:


data1['Solar'] = data1['Ozone'].fillna(mean_ozone)
data1.isnull().sum()


# In[ ]:




