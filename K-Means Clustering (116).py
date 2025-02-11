#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import seaborn as sns


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


Univ.isnull().sum()


# In[5]:


Univ.describe()


# In[6]:


fig, axes = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=Univ["Top10"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Top10 Levels")

sns.histplot(Univ["Top10"], kde=True, ax=axes[1], color="blue", bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Top10 Levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[7]:


# REad all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[8]:


Univ1


# In[9]:


cols=Univ1.columns
cols


# In[10]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1), columns=cols)
scaled_Univ_df
# scaler.fit_transform(Univ1)


# In[11]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1), columns=cols)
scaled_Univ_df
# scaler.fit_transform(Univ1)


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[13]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[14]:


clusters_new.labels_


# In[15]:


set(clusters_new.labels_)


# In[16]:


Univ['clusterid_new'] = clusters_new.labels_


# In[17]:


Univ


# In[18]:


Univ.sort_values(by="clusterid_new")


# In[19]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# In[20]:


Observations:
- Custer 2 appears to be top rated universities cluster as the cluster as the cut off score,Top10,SFRatio parameter mean values are 
-cluster 1 appear to occupy the middle level rated universities
- cluster 0 comes as the lowerlevel rated universities


# In[21]:


wcss = []
for i in range(1,20):
    
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    #kmeans.fit(Univ1)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


Observations: 
-for the above graph elbow join that is the rte of change of slow decreases

