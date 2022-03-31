#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv("Data.csv")
X=df.iloc[ : , :-1 ].values
Y=df.iloc[: , -1].values


# In[11]:


df.head()


# In[9]:


print(X)


# In[10]:



print(Y)


# In[12]:


df.isnull().sum()


# In[15]:


df['Age'].mean()


# In[16]:


df['Age']=df['Age'].fillna(38.777)


# In[17]:


df.isnull().sum()


# In[18]:


df.head()


# In[19]:


df.head(20)


# In[20]:


df['Salary'].mean()


# In[21]:


df['Salary']=df['Salary'].fillna(63777.77777)


# In[22]:


df.isnull().sum()


# In[23]:


df.head(20)


# In[ ]:




