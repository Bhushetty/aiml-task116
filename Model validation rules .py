#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Evaluate using a train a testset
from pandas import read_csv
from sklearn.model_selection import tran_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[ ]:


data = read_csv("diabetes.csv")
data


# ### Model validation using train_test_split()

# In[ ]:


#split the data into train test sets and find the test accuracy
array = data.values
x = array[:,0:8]
y = array[:,8]
X_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 3)
model1 = decisionTreeClassifier()
model1.fit(x_train,y_train)
y_predict = model1.predict(x_test)
print(classifaction_report(y_test,y_predict))


# ### Evaluate using K-fold cross validation

# In[ ]:


#Evaluate using cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


x = array[:,0:8]
y = array[:, 8]
#num_folds = 10
#seed = 7
kfold = KFold(n_splits=7)
model2 = DecisionTreeClassifier()
results2 = cross_val_score(model2, X, Y, cv=kfold)
print(results2)

