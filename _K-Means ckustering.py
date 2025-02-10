#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
Univ = pd.readimpo_csv("Universities.csv")
Univ


# In[5]:


Univ.info()


# In[ ]:


Observations:
-There is no null values.


# In[6]:


Univ.describe()


# In[8]:


#Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[9]:


Univ1


# In[10]:


Univ1.columns


# In[13]:


cols = Univ1.columns


# In[14]:


#standarization function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df
#scaler.fit_transform(Univ1)


# In[ ]:




