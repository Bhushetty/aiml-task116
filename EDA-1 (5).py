#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


#printing the information
data.info()


# In[4]:


#Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)

data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1
# In[5]:


data1 = data.drop(['Unnamed: 0','Temp C'], axis =1)
data1


# In[6]:


data['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[8]:


data1.drop_duplicates(keep='first',inplace = True)
data1


# In[9]:


#RENAMING THE COLUMNS
data1.rename({'Solar.R': 'Solar'},axis=1,inplace = True)
data1


# In[10]:


#Impute the missing value in the table


# In[11]:


data1.info()


# In[12]:


#Display data1 missing values count in each colimn using isnull().sum()
data1.isnull().sum()


# In[13]:


#visualize data1 missing value using heat map
cols = data1.columns
colors = ['black','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[15]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


data1['Solar'] = data1['Ozone'].fillna(mean_ozone)
data1.isnull().sum()


# In[17]:


#Find the mode values of categorical column(weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[18]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[19]:


#Find the mode values of categorical column(weather)
print(data1["Month"].value_counts())
mode_weather = data1["Month"].mode()[0]
print(mode_weather)


# In[20]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# In[21]:


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

# In[22]:


#create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')


# In[23]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert= False)


# In[24]:


data1["Ozone"].describe()


# In[25]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[26]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm",plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[27]:


# observations from Q-Q plot
.The data does not follow normal distribution as the points are deviating significantly away from the red 
.The data shows a right skewed distribution and possible outliers


# In[28]:


# create a figure for violen plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("violin plot")


# # Other visualizations that could help  understand the data
# 

# In[29]:


#create a figure for violin plot
sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("Violin plot")
#show the plot
plt.show()


# In[30]:


sns.swarmplot(data=data1,x ="Weather",y = "Ozone",color="orange",palette="Set2",size=6)


# In[31]:


sns.stripplot(data=data1, x = "Weather",y = "Ozone",color="Orange",palette="Set1",size=6,jitter = True)


# In[32]:


sns.kdeplot(data=data1["Ozone"],fill=True,color="blue")
sns.rugplot(data=data1["Ozone"],color="black")


# In[33]:


#category wise boxplot for ozone
sns.boxplot(data = data1, x = "Weather",y="Ozone")


# # Corerelation coefficient and pair plots

# In[34]:


plt.scatter(data1["Wind"],data1["Temp"])


# In[35]:


# Comoute pearson corerelation coefficient
data1["Wind"].corr(data1["Temp"])


# In[36]:


#read all numeric columns into a new table
data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




