#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Data1.csv")


# In[4]:


df.isnull().sum()


# In[5]:


df.head()


# In[6]:


X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


lr=LinearRegression()


# In[10]:


lr.fit(X,Y)


# In[16]:


y_test_pred=lr.predict(X_test)


# In[17]:


y_test_pred


# In[18]:


df1=Y_test.copy()


# In[19]:


df1


# In[21]:


y_test_pred=y_test_pred.reshape(len(y_test_pred),1)


# In[24]:


df2=y_test_pred.copy()


# In[29]:


from sklearn.metrics import r2_score


# In[30]:


r2_score(Y_test,y_test_pred)


# In[ ]:




