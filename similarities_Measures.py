#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.spatial.distance import cosine,euclidean
from scipy.stats import pearsonr


# In[3]:


user1 = np.array([4,5,2,3,4])
user2 = np.array([5,3,2,4,5])


# In[4]:


cosine_similarity = 1 - cosine(user1,user2)
print(f"Cosine Similarity: {cosine_similarity:4f}")


# In[10]:


pearson_corr, _ =pearsonr(user1, user2)
print(f"Pearson Correalation Similarity: {pearson_corr:.4f}")
print(_)


# In[11]:


euclidean_distance = euclidean(user1,user2)

euclidean_similarity = 1 / (1 + euclidean_distance)
print(f"Euclidean Distance Similarity: {euclidean_similarity:.4f}")


# In[ ]:




