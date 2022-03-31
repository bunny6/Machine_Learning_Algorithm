#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv('Salary_Data.csv')


# In[5]:


df.head(50)


# In[6]:


df.isnull().sum()


# In[7]:


X=df.drop('Salary',axis=1)
Y=df[['YearsExperience']]


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


regression=LinearRegression()


# In[29]:


regression.fit(X_train,Y_train)


# In[30]:


y_train_pred=regression.predict(X_train)


# In[31]:


y_test_pred=regression.predict(X_test)


# In[32]:


plt.scatter(X_train,Y_train,color="Red")
plt.plot(X_train,y_train_pred,color="green")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary and Experience")
plt.show()


# In[25]:


plt.scatter(X_test,Y_test,color="Red")
plt.plot(X_test,y_test_pred,color="green")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary and Experience")
plt.show()


# In[ ]:




