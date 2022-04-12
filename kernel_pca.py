#!/usr/bin/env python
# coding: utf-8

# # Kernel PCA

# ## Importing the libraries

# In[2]:


import numpy as np                       #importing the libraries
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[3]:


dataset = pd.read_csv('Wine.csv')       #importing the dataset.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[4]:


from sklearn.model_selection import train_test_split              #spliting the data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Feature Scaling

# In[5]:


from sklearn.preprocessing import StandardScaler             #doing the scaling on dataset.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Applying Kernel PCA

# In[6]:


from sklearn.decomposition import KernelPCA             #importing and applying kernal. kernal is radical basis function
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


# ## Training the Logistic Regression model on the Training set

# In[7]:


from sklearn.linear_model import LogisticRegression     #training model: Logistic regression. 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# ## Making the Confusion Matrix

# In[8]:


from sklearn.metrics import confusion_matrix, accuracy_score  #accuracy of the model.
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# ## Visualising the Training set results

# In[9]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# ## Visualising the Test set results

# In[10]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

