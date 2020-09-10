#!/usr/bin/env python
# coding: utf-8

# # Time Series Analysis
#   
#   ## Predicting Milk Production

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[4]:


tsDF = pd.read_csv('./data/monthly_milk_production.csv')
tsDF['midx'] = tsDF['Month'].str[-2:]

tsDF.head(20)


# ## Adding the month period column

# In[5]:


tsDF['period'] = range(1,len(tsDF)+1)
tsDF.head(20)


# In[6]:


## Identifying the trending line
X = tsDF['period'].iloc[:].values.reshape(-1, 1)  # values converts it into a numpy array
Y = tsDF['Production'].iloc[:].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

# Instantiate a Linear Regression Object
lr_model = LinearRegression()

# Train the model
lr_model.fit(X, Y)

#
y_prediction = lr_model.predict(X)  # make predictions


# In[7]:


tsDF['lc'] = y_prediction


# In[8]:


tsDF.head(15)


# In[9]:


with plt.style.context('ggplot'):
    plt.figure(figsize=(8,6))
    plt.plot(tsDF['Production'], label='Production')
    plt.plot(tsDF['lc'], label='Linear Component')
    plt.legend(loc=2)


# In[10]:


tsDF['detrended'] = tsDF['Production']-tsDF['lc']


# In[11]:


with plt.style.context('ggplot'):
    plt.figure(figsize=(8,6))
    plt.plot(tsDF['detrended'], label='Production Detrended')
    plt.legend(loc=2)


# ## Calculating the Seasonal Component

# In[12]:


# Calculate the seasonal component and remove it from the de-trended line
mavgDF = tsDF.groupby(['midx']).mean()
mavgDF = pd.concat([mavgDF]*13)['detrended']


# In[13]:


tsDF['scomponent'] = mavgDF.values


# In[14]:


tsDF.head(20)


# In[15]:


tsDF['error'] = tsDF['detrended']-tsDF['scomponent']


# In[16]:


with plt.style.context('ggplot'):
    plt.figure(figsize=(8,6))
    plt.plot(tsDF['error'], label='Error Component')
    plt.legend(loc=2)


# In[17]:


tsDF['prediction'] = tsDF['lc']+tsDF['scomponent']


# In[18]:


with plt.style.context('ggplot'):
    plt.figure(figsize=(8,6))
    plt.plot(tsDF['Production'], label='Production')
    plt.plot(tsDF['prediction'], label='prediction')
    plt.legend(loc=2)


# In[ ]:





# In[ ]:





# In[ ]:




