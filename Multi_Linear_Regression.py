#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[35]:


df=pd.read_csv("50_Startups.csv")


# In[36]:



X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[37]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[38]:


ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))


# In[39]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


lr=LinearRegression()


# In[42]:


lr.fit(X_train,Y_train)


# In[43]:


Y_train_pred=lr.predict(X_train)


# In[44]:


print(np.concatenate((Y_train_pred.reshape(len(Y_train_pred),1), Y_train.reshape(len(Y_train),1)),1))


# In[45]:


y_test_pred = lr.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_test_pred.reshape(len(y_test_pred),1), Y_test.reshape(len(Y_test),1)),1))


# In[ ]:




