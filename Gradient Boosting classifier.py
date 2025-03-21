#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


#Load dataset
df = pd.read_csv('diabetes.csv')
df


# In[3]:


#Feature and target
X = df.drop('class',axis=1)
y = df['class']

#Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[4]:


#perform train,test split on the dataset
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.8,random_state = 42)


# In[5]:


#Instantiate the model an define the parameters
gbc = GradientBoostingClassifier(random_state=42)

#set up KFold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

#Hyperparamter grid
param_grid = {
    'n_estimators':[50,100,150],
    'learning_rate': [0.01,0.1,0.2],
    'max_depth': [3,4,5],
    'subsample': [0.8,1.0]
}
#grid search with cross-validation
grid_search = GridSearchCV(estimator=gbc,
                          param_grid=param_grid,
                          cv=kfold,
                          scoring='recall',
                          n_jobs=-1,
                          verbose=1)


# In[ ]:


#Fit the model
grid_search.fit(X_train,y_train)
#Best paramaters and score
print("Best Parameters:",grid_search.best_params_)
print("Best Cross-Validation Recall:",grid_search.best_score_)


# In[9]:


#Evaluate on test data using best estimtor
best_model = grid_search.best_estimators_
y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassifiaction Report:\n",classification_report(y_test, y_pred))


# ###Identify feature importance scores using XGBClassifer
# 

# In[10]:


best_model.feature_importances_


# In[ ]:




