#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv("Universities.csv")
df


# In[6]:


#mean vakue of sat score
np.mean(df["SAT"])


# In[7]:


#median of data
np.median(df["SAT"])


# In[8]:


#standard deviation of data
np.std(df["GradRate"])


# In[9]:


#Find the variance
np.var(df["SFRatio"])


# In[10]:


df.describe()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


plt.figure(figsize=(4,6))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[14]:


s = [20,15,10,30,25,35,28,40,60]
scores = pd.Series(s)
scores


# In[15]:


plt.boxplot(scores,vert=False)


# In[16]:


plt.boxplot(scores)


# In[17]:


scoress = [20,15,10,30,25,35,28,40,60,120,150]
scores = pd.Series(s)


# In[20]:


plt.boxplot(scores,vert = False)


# #identify outliers in universities dataset

# In[26]:


df = pd.read_csv("universities.csv")
df


# In[29]:


plt.figure(figsize=(6,2))
plt.title("Box plot for SAT Score")
plt.boxplot(df["SAT"], vert = False)


# In[ ]:





# In[ ]:





# In[ ]:




