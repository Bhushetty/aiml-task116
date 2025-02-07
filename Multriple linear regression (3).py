#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[52]:


#Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[53]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()
                                   


# In[54]:


Description of columns(mile per Gallon)
-MpG:meilege of the car(X1 column)
-HP: horse power of the car(X2 column)
-vol:Volume of the car(X3 column)
-Sp : Top speed of the car 
-WT: weightned of the car


# In[55]:


Assumption in Multilinear Regression 
1.Linearity:The relationship between the predictors(x)and the response (y) is linear.
2.Independence:Observations are independent of each other.
3.HomoScedasticity:The residuals exhibit const varience at all levels at all levels of the predictor
4.Normal Distribution of Errors: The residuals of the model are normally distributed.
5.No multicollinearity:The independence     


# #EDA

# In[56]:


cars.info()


# In[57]:


cars.isna().sum()


# # Observations about info(),missing values
# There are no missing values
# There are 81 observations(81 different cars data)
# The data types of the columns are also relevant and valid

# In[58]:


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


# In[59]:


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


# In[60]:


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


# In[61]:


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


# Observation from boxplot and histograms
# - There are some extreme values  observed in towrds the right tail of sp and HP distribution'
# - In VOL and WT columns,a few outliers are observed in both tails of their distribution
# -The extreme values of cars data may have come from the specially designed nature of cars 
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered wihile building the regression model 

# Checking for duplicated rows
# 

# In[62]:


cars[cars.duplicated()]


# In[63]:


#Pair lots and correlation coefficent
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[64]:


cars.corr()


# Observations from correlation plots and coefficient 
# -Between x and y,all the x variables are showing moderate to high   correlation strength,highest being between HP and MPG
# -Therefore this dataset qualifies for building linear regression model to predict MPG
# -Among x column(x1,x2,x3and x4),some very high correlation strngth are observed between SP vs HP,VOL vs WT
# -The high correlation among x column is not desirable at it might lead to multicollinearity probelm
# 

# In[65]:


#build model
#import satatsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[66]:


model1.summary()


# Observation for model summary:
# - The R-squared and adjusted R-squared values are good and about 75% of variablity in Y is explained by x columns
# -The probability value with respect to F-statstic is close to zero,indicating that all of someof X column are significant
# -The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves,which nedd to be furthur explore

# Performance metrices for model1

# In[67]:


#Find the performance metrics
#Create a data frame with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[68]:


#predict for the given X data column
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[69]:


#FInd the performance metrices
#create a data frame with actual y and predicted y and predicted y colums
 
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[70]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head


# In[72]:


#compute the mean squared error(MSE),RmSe for model1
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE:", mse)
print("RMSE :",np.sqrt(mse))


# Checking for multicollinearity among X-cloumns using VIF method
# 
# 

# In[73]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# Observations for VIF values:
#  - The ideal range of VIF values shall be between 0 to 10.However sightly higher values can be tolerated
#  -As seen from the very high VIF values for vol and wt,it is clear that they are prone to multicollinearity problem
#  -Hence it is decided to drop one of the column(either VOL or WT)to overcome the multicollinearity.
#  -It is decided to drop WT and retain VOL column in further models

# In[74]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[ ]:


#build model2 
model2 = smf.ols('MPG~'HP+Vol+SP'data=cars',)
model2.summary()


# In[77]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[78]:


model2.summary()


# Performance metrics for model2
# #Find the performance metirces
# #Create a data frame with actual y and predicted y column'
# df2 = pd.DataFrame()
# df2["actual_y2"] = cars["MPG"]
# df2.head()

# In[80]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[81]:


#predict for the given x data coolumns
pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[ ]:


from sklearn.metices import mean _squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :")


# In[ ]:


oBSERVATION
The adjusted R-squared value imporived slightly to 0.76
All the p-values for model paramaters are less than 5% hence they are significant
Therefore the hp,vol,sp columns are finialized as the significant predictor for the Mpg response variable 
There is no improvement in MSE value

