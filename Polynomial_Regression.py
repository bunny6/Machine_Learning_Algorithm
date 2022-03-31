#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


df=pd.read_csv("Position_Salaries.csv")


# In[3]:


df.head()


# In[4]:


X=df.iloc[:,1:-1].values
Y=df.iloc[:,-1].values


# In[6]:


lr=LinearRegression()
lr.fit(X,Y)


# In[8]:


pr=PolynomialFeatures(degree=2)
X_poly=pr.fit_transform(X)
lr2=LinearRegression()
lr2.fit(X_poly,Y)


# In[11]:


x2=lr.predict(X)


# In[19]:


x3=lr2.predict(X_poly)


# In[16]:


plt.scatter(X,Y,color='red')
plt.plot(X,x2,color='Blue')


# In[ ]:


plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_re.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[23]:


plt.scatter(X,Y,color='red')
plt.plot(X,lr2.predict(pr.fit_transform(X)),color="blue")


# In[ ]:




