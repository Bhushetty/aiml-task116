#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[5]:


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df


# In[6]:


x = df.drop(columns=['target'])
y = df['target']


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)


# In[10]:


predictions = clf.predict(x_test)
accuracy = accuracy_score(y_test, predictions)


# In[11]:


# Streamlit UI
st.title("Decision Tree Classifier - Iris Dataset")
st.write(f"Model Accuracy: {accuracy:.2f}")

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length", float(df.iloc[:, 0].min()), float(df.iloc[:, 0].max()), float(df.iloc[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(df.iloc[:, 1].min()), float(df.iloc[:, 1].max()), float(df.iloc[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(df.iloc[:, 2].min()), float(df.iloc[:, 2].max()), float(df.iloc[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(df.iloc[:, 3].min()), float(df.iloc[:, 3].max()), float(df.iloc[:, 3].mean()))


# In[13]:


#prdictions
data-input = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
prediction = clf.predict(data_input)
prediction_class = data.target_names[prediction[0]]

st.subheader("prediction")
st.write(f"Predicted Class: **{predicted_class}**")


# In[14]:


if st.button("Show Decision Tree Diagram"):
    fig, ax = plt.subplots(figsize=(12,8))
    tree.plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Decision Tree Visualization", use_column_width=True)


# In[ ]:




