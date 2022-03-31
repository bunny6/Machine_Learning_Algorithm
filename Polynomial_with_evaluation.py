#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv('Data1.csv')
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


# In[5]:


df.head()


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[14]:


poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X_train)
lr=LinearRegression()


# In[16]:


lr.fit(X_poly,Y_train)


# In[17]:


y_pred=lr.predict(poly.transform(X_test))


# In[29]:


df2=Y_test.copy()


# In[22]:


from sklearn.metrics import r2_score


# In[23]:


r2_score(Y_test,y_pred)


# In[30]:


y_pred.shape


# In[ ]:




