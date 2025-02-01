#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[5]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()                    


# In[6]:


data1.info()


# In[7]:


data1.isnull().sum()


# In[8]:


data1.describe()


# In[10]:


#Box plot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily sales")
plt.boxplot(data1["daily"], vert = False)
plt.show


# In[11]:


sns.histplot(data1['daily'],kde = True,stat='density',)
plt.show()


# In[ ]:


#observations
- There are no missing values
- The daily column values appears to be right-skewed
- The sunday column values also appear to be skewed
-There are two outliers in both column and also in sunday column as observed from the


# In[13]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[14]:


data1["daily"].corr(data1["sunday"])


# In[15]:


data1[["daily","sunday"]].corr()


# In[16]:


data1.corr(numeric_only=True)


# #observations on corelation strength
# - The relationship between x(daily)and y(sunday)is seen to be linear as seen from scatter plot
# -The correlation is strong and postive with pearsons correlation coefficient of 0.958154

# In[19]:


#Fit a Linear regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[20]:


model1.summary()


# In[ ]:




