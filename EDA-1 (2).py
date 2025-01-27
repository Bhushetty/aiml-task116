#!/usr/bin/env python
# coding: utf-8

# In[20]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[22]:


#printing the information
data.info()


# In[23]:


#Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)

data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1
# In[24]:


data1 = data.drop(['Unnamed: 0','Temp C'], axis =1)
data1


# In[25]:


data['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[26]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[27]:


data1.drop_duplicates(keep='first',inplace = True)
data1


# In[28]:


#RENAMING THE COLUMNS
data1.rename({'Solar.R': 'Solar'},axis=1,inplace = True)
data1


# In[29]:


#Impute the missing value in the table


# In[30]:


data1.info()


# In[31]:


#Display data1 missing values count in each colimn using isnull().sum()
data1.isnull().sum()


# In[32]:


#visualize data1 missing value using heat map
cols = data1.columns
colors = ['black','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[33]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[34]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[35]:


data1['Solar'] = data1['Ozone'].fillna(mean_ozone)
data1.isnull().sum()


# In[37]:


#Find the mode values of categorical column(weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[38]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[39]:


#Find the mode values of categorical column(weather)
print(data1["Month"].value_counts())
mode_weather = data1["Month"].mode()[0]
print(mode_weather)


# In[40]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# In[45]:


fig,axes = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})

#plot the boxplot in the first(top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient =' h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

#plot the histogram with kde curve in the second (bottom) subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")

#Adjust layour for better spacing
plt.tight_layout()

#show the plot
plt.show()


# OBSERVATIONS
# . The ozone column has extreme values beyond 81 as seen from box plot
# . The same is confirmed from the below right-skewed histogram

# In[46]:


#create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')


# In[ ]:




