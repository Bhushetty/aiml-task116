#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[5]:


data1.info()


# In[7]:


data1.describe()


# In[11]:


sns.boxplot(data=data1['daily'], color='orange',width=0.5,orient='h')


# In[13]:


sns.histplot(data=data1['daily'],kde=True, color='blue',bins=30)


# In[15]:


sns.scatterplot(data = data1, x = "daily", y="sunday")


# In[ ]:




