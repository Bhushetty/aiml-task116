#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[6]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()                    


# In[7]:


data1.info()


# In[8]:


data1.isnull().sum()


# In[9]:


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


# In[12]:


#observations
- There are no missing values
- The daily column values appears to be right-skewed
- The sunday column values also appear to be skewed
-There are two outliers in both column and also in sunday column as observed from the


# In[ ]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[13]:


data1["daily"].corr(data1["sunday"])


# In[14]:


data1[["daily","sunday"]].corr()


# In[15]:


data1.corr(numeric_only=True)


# #observations on corelation strength
# - The relationship between x(daily)and y(sunday)is seen to be linear as seen from scatter plot
# -The correlation is strong and postive with pearsons correlation coefficient of 0.958154

# In[16]:


#Fit a Linear regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[17]:


model1.summary()


# In[ ]:


Interpretation:
    -r squared = 1-perfect fit(all variance explained)
    -R squared=0-Model does not explain any variance
    -R squared close to 1 - Good model fit
    -Rsquared close to 0 - poor model fit


# In[18]:


#plot the scatter plot and overlay the fitted stright line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x,y,color = 'm', marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
#prdicted response vector
y_hat = b0 + b1*x
#plotting the regression vector
plt.plot(x,y_hat,color = 'g')
#putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:


#observation for model summary
-The probability(p-value) for intercept(beta_0)is 0.707 > 0.05
-Therefore the intercept coefficent may not be that significant in prediction
- However the p-Vlaue for "daily"(beta_1) is 0.00 < 0.05
-Therefore the beta_1 coefficient is highly signicant and is contributint to prediction.


# In[19]:


#print the fitted line coefficients(Beta-0 and beta-1)
model1.params


# In[20]:


print(f'model t-values:\n{model1.tvalues}\n--------------\nmodel p-values:\n{model1.pvalues}')


# In[21]:


model1.rsquared,model1.rsquared_adj


# In[22]:


#predict for 200 and 300 dailt circulation
newdata=pd.Series([200,300,1500])


# In[24]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[25]:


model1.predict(data_pred)


# In[27]:


#predict on all given training data
pred = model1.predict(data1["daily"])
pred


# In[28]:


#predicted on all values as a  column in data1
data1["Y_hat"] = pred
data1


# In[29]:


#Compute the error values (residuals)and add as another column
data1["residuals"]=data1["sunday"]-data1["Y_hat"]
data1


# In[30]:


#compute Mean squared error for the model
mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ", rmse)


# In[ ]:




