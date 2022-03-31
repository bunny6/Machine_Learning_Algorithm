#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv("Data1.csv")
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


# In[4]:


df.head()


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[7]:


from sklearn.tree import DecisionTreeRegressor


# In[8]:


dt=DecisionTreeRegressor(random_state=1)


# In[9]:


dt.fit(X_train,Y_train)


# In[11]:


Y_test_pred=dt.predict(X_test)


# In[13]:


from sklearn.metrics import r2_score


# In[14]:


r2_score(Y_test,Y_test_pred)


# In[ ]:




