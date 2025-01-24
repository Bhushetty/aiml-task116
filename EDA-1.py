#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[6]:


#printing the information
data.info()


# In[7]:


#Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)

data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1
# In[8]:


data1 = data.drop(['Unnamed: 0','Temp C'], axis =1)
data1


# In[10]:


data['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[11]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[12]:


data1.drop_duplicates(keep='first',inplace = True)
data1


# In[ ]:




