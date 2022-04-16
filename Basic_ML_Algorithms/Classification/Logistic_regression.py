#importing the libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset.
df=pd.read_csv("Social_Network_Ads.csv")
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

df.head()

#Spliting the data in train and test.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)

#Scaling the data.
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#importing the logistic regression. 
from sklearn.linear_model import LogisticRegression

#training the model on training dataset.
lr=LogisticRegression(random_state=1)
lr.fit(X_train,Y_train)

#predicting on test dataset.
Y_test_pred=lr.predict(X_test)

#Printing the actual and the predicted data side by side.
print(np.concatenate((Y_test_pred.reshape(len(Y_test_pred),1),Y_test.reshape(len(Y_test),1)),1))

#Importing performance metrics.
from sklearn.metrics import confusion_matrix,accuracy_score

#Checking the performance of the model by Creating Confusion matrix and printing  accuracy score.
cm=confusion_matrix(Y_test,Y_test_pred)
print(cm)
accuracy_score(Y_test,Y_test_pred)






