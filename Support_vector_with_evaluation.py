#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("Data1.csv")
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


# In[9]:


Y


# In[11]:


Y=Y.reshape(len(Y),1)


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[13]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X_train=sc_X.fit_transform(X_train)
Y_train=sc_y.fit_transform(Y_train)


# In[5]:


from sklearn.svm import SVR


# In[14]:


sv=SVR(kernel='rbf')


# In[15]:


sv.fit(X_train,Y_train)


# In[ ]:


y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[17]:


y_pred=sc_y.inverse_transform(sv.predict(sc_X.fit_transform(X_test)))


# In[18]:


from sklearn.metrics import r2_score


# In[19]:


r2_score(Y_test,y_pred)


# In[ ]:




