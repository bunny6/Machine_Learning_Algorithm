#importing the libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset.

df=pd.read_csv('Data1.csv')
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

df.head()

#Spliting the data in train and test.

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


#importing the logistic regression and polynomial regression.

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



#Defining the polynomial feature. 

poly=PolynomialFeatures(degree=4)    #Giving degree value = 4. Polynomial function will have value = 4.
X_poly=poly.fit_transform(X_train)
lr=LinearRegression()


#Predicting on training dataset.

lr.fit(X_poly,Y_train)


#Predicting on Test Dataset.

y_pred=lr.predict(poly.transform(X_test))


#Copying the data of y_test into df2.
df2=Y_test.copy()

#importing the performance metrics.

from sklearn.metrics import r2_score

#checking the performance on test dataset.

r2_score(Y_test,y_pred)








