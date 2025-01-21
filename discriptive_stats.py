#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Universities.csv")
df


# In[4]:


#mean vakue of sat score
np.mean(df["SAT"])


# In[5]:


#median of data
np.median(df["SAT"])


# In[6]:


#standard deviation of data
np.std(df["GradRate"])


# In[7]:


#Find the variance
np.var(df["SFRatio"])


# In[8]:


df.describe()


# In[ ]:




