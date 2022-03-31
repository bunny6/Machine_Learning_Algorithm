#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


df=pd.read_csv("Social_Network_Ads.csv")
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


# In[6]:


df.head()


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


lr=LogisticRegression(random_state=1)
lr.fit(X_train,Y_train)


# In[14]:


Y_test_pred=lr.predict(X_test)


# In[19]:


print(np.concatenate((Y_test_pred.reshape(len(Y_test_pred),1),Y_test.reshape(len(Y_test),1)),1))


# In[20]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[23]:


cm=confusion_matrix(Y_test,Y_test_pred)
print(cm)
accuracy_score(Y_test,Y_test_pred)


# In[ ]:




