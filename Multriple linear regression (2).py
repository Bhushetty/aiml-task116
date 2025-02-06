#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[18]:


#Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[19]:


Description of columns(mile per Gallon)
-MpG:meilege of the car(X1 column)
-HP: horse power of the car(X2 column)
-vol:Volume of the car(X3 column)
-Sp : Top speed of the car 
-WT: weightned of the car


# In[20]:


Assumption in Multilinear Regression 
1.Linearity:The relationship between the predictors(x)and the response (y) is linear.
2.Independence:Observations are independent of each other.
3.HomoScedasticity:The residuals exhibit const varience at all levels at all levels of the predictor
4.Normal Distribution of Errors: The residuals of the model are normally distributed.
5.No multicollinearity:The independence     


# #EDA

# In[9]:


cars.info()


# In[10]:


cars.isna().sum()


# # Observations about info(),missing values
# There are no missing values
# There are 81 observations(81 different cars data)
# The data types of the columns are also relevant and valid

# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[15]:


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


# In[16]:


Observation from boxplot and histograms
- There are some extreme values  observed in towrds the right tail of sp and HP distribution'
- In VOL and WT columns,a few outliers are observed in both tails of their distribution
-The extreme values of cars data may have come from the specially designed nature of cars 
- As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered wihile building the regression model 


# Checking for duplicated rows
# 

# In[ ]:


cars[cars.duplicated()]


# In[4]:


#Pair lots and correlation coefficent
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[5]:


cars.corr()


# Observations from correlation plots and coefficient 
# -Between x and y,all the x variables are showing moderate to high   correlation strength,highest being between HP and MPG
# -Therefore this dataset qualifies for building linear regression model to predict MPG
# -Among x column(x1,x2,x3and x4),some very high correlation strngth are observed between SP vs HP,VOL vs WT
# -The high correlation among x column is not desirable at it might lead to multicollinearity probelm
# 

# In[25]:


#build model
#import satatsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[26]:


model1.summary()


# Observation for model summary:
# - The R-squared and adjusted R-squared values are good and about 75% of variablity in Y is explained by x columns
# -The probability value with respect to F-statstic is close to zero,indicating that all of someof X column are significant
# -The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves,which nedd to be furthur explore

# Performance metrices for model1

# In[27]:


#Find the performance metrics
#Create a data frame with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[28]:


#predict for the given X data column
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[29]:


#FInd the performance metrices
#create a data frame with actual y and predicted y and predicted y colums
 
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[ ]:




