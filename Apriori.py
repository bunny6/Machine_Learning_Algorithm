#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().system('pip install apyori')


# In[9]:


df=pd.read_csv("Market_Basket_Optimisation.csv", header= None)


# In[10]:


df.head()


# In[11]:


df.describe()


# In[7]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df.shape


# In[15]:


transactions=[]
for i in range(0,7501):
    transactions.append([str(df.values[i,j])for j in range(0,20)])


# In[16]:


from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


# In[17]:


result=list(rules)


# In[18]:


result


# In[20]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in result]
    rhs         = [tuple(result[2][0][1])[0] for result in result]
    supports    = [result[1] for result in result]
    confidences = [result[2][0][2] for result in result]
    lifts       = [result[2][0][3] for result in result]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(result), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# In[21]:


resultsinDataFrame


# In[22]:


resultsinDataFrame.nlargest(n = 10, columns = 'Lift')


# In[ ]:




