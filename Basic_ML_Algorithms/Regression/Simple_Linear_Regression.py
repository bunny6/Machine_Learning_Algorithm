#importing the libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset.
df=pd.read_csv('Salary_Data.csv')

df.head(50)

#checking for null values in dataset.
df.isnull().sum()

#Droping the last column and saving other columns in X, and taking the last column as Y.
X=df.drop('Salary',axis=1)
Y=df[['YearsExperience']]

#Spliting the data in train and test.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

#importing the linear regression.
from sklearn.linear_model import LinearRegression
regression=LinearRegression()

#training the model on training dataset.
regression.fit(X_train,Y_train)

#predicting on training dataset.
y_train_pred=regression.predict(X_train)

#predicting on test dataset.
y_test_pred=regression.predict(X_test)

#Plotting the X_train and y_train_pred.
plt.scatter(X_train,Y_train,color="Red")
plt.plot(X_train,y_train_pred,color="green")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary and Experience")
plt.show()


#plotting the X_test,y_test_pred.
plt.scatter(X_test,Y_test,color="Red")
plt.plot(X_test,y_test_pred,color="green")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary and Experience")
plt.show()






