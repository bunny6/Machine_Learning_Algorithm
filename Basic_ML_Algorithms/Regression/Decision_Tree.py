#importing the libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset and spliting it into X and Y.
df=pd.read_csv("Data1.csv")
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

df.head()

#Spliting the data into Train and test.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

#Importing Decision tree.
from sklearn.tree import DecisionTreeRegressor

#Defining the Decision Tree.
dt=DecisionTreeRegressor(random_state=1)

#Training the model on training dataset.
dt.fit(X_train,Y_train)

#Predicting on the test data.
Y_test_pred=dt.predict(X_test)

#importing the r2_score.
from sklearn.metrics import r2_score

#checking model performance.
r2_score(Y_test,Y_test_pred)





