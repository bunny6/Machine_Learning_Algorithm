#importing the dataset.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset and spliting the data into X and Y.

df=pd.read_csv("50_Startups.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#importing ColumnTransformer and OneHotEncoding for Encoding.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Encoding the data by using OneHotEncoder.

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#Spliting the data into train and test.

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)


#Importing linear regression.

from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# Training the model on training dataset.

lr.fit(X_train,Y_train)

#Predicting on the training data.

Y_train_pred=lr.predict(X_train)

#Printing the actual and the predicted data side by side for training dataset.

print(np.concatenate((Y_train_pred.reshape(len(Y_train_pred),1), Y_train.reshape(len(Y_train),1)),1))

#predicting on test dataset.

y_test_pred = lr.predict(X_test)

##Checking the performance of the model by Creating Confusion matrix and printing  accuracy score
np.set_printoptions(precision=2)
print(np.concatenate((y_test_pred.reshape(len(y_test_pred),1), Y_test.reshape(len(Y_test),1)),1))






