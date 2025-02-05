#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[5]:


#Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[ ]:


Description of columns(mile per Gallon)
-MpG:meilege of the car(X1 column)
-HP: horse power of the car(X2 column)
-vol:Volume of the car(X3 column)
-Sp : Top speed of the car 
-WT: weightned of the car


# In[ ]:


Assumption in Multilinear Regression 
1.Linearity:The relationship between the predictors(x)and the response (y) is linear.
2.Independence:Observations are independent of each other.
3.HomoScedasticity:The residuals exhibit const varience at all levels at all levels of the predictor
4.Normal Distribution of Errors: The residuals of the model are normally distributed.
5.No multicollinearity:The independence     


# #EDA

# In[6]:


cars.info()


# In[7]:


cars.isna().sum()


# # Observations about info(),missing values
# There are no missing values
# There are 81 observations(81 different cars data)
# The data types of the columns are also relevant and valid

# In[15]:


#Create a figure wth two subplots
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
#creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
#Creating a histogrm in the same x-axis
sns.histplot(data=cars, x='HP',ax=ax_hist, bins=30, kde=True,stat="density")
ax_hist.set(ylabel='Density')
#Adjust layout
plt.tight_layout()
plt.show()


# In[16]:


#Create a figure wth two subplots
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
#creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
#Creating a histogrm in the same x-axis
sns.histplot(data=cars, x='SP',ax=ax_hist, bins=30, kde=True,stat="density")
ax_hist.set(ylabel='Density')
#Adjust layout
plt.tight_layout()
plt.show()


# In[17]:


#Create a figure wth two subplots
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
#creating a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
#Creating a histogrm in the same x-axis
sns.histplot(data=cars, x='VOL',ax=ax_hist, bins=30, kde=True,stat="density")
ax_hist.set(ylabel='Density')
#Adjust layout
plt.tight_layout()
plt.show()


# In[18]:


#Create a figure wth two subplots
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
#creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
#Creating a histogrm in the same x-axis
sns.histplot(data=cars, x='WT',ax=ax_hist, bins=30, kde=True,stat="density")
ax_hist.set(ylabel='Density')
#Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:


Observation from boxplot and histograms
- There are some extreme values  observed in towrds the right tail of sp and HP distribution'
- In VOL and WT columns,a few outliers are observed in both tails of their distribution
-The extreme values of cars data may have come from the specially designed nature of cars 
- As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered wihile building the regression model 


# Checking for duplicated rows
# 

# In[19]:


cars[cars.duplicated()]


# In[21]:


#Pair lots and correlation coefficent
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[22]:


cars.corr()


# In[ ]:




