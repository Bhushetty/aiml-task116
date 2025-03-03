#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


# iris = datasets.load_iris(as_frame=True).frame


# In[3]:


iris = pd.read_csv("iris.csv")


# In[4]:


iris


# In[5]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[6]:


iris.info()


# In[7]:


iris[iris.duplicated(keep=False)]


#  ### perform label encoding of target column
#  

# In[8]:


#Encode the three flower classes as 0,1,2
labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[9]:


iris.info()


# In[10]:


# Convert the target column data to interger 
iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[11]:


iris.head(3)


# In[12]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[13]:


Y


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state = 1)
x_train


# # Building Decision Tree Classifier using Entropy criteria

# In[15]:


model = DecisionTreeClassifier (criterion = 'entropy',max_depth = None)
model.fit(x_train,y_train)


# In[16]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[17]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor','virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[18]:


#predicting on test data
preds = model.predict(x_test)# predicting on test data set
preds


# In[20]:


print(classification_report(y_test,preds))


# In[ ]:




