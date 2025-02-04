#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[4]:


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




