#importing libraries.
import numpy as np                                
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset.
dataset = pd.read_csv('Data.csv')               
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training set and Test set.
from sklearn.model_selection import train_test_split                       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training XGBoost on the Training set.
from xgboost import XGBClassifier                           
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Checking the performance of the model by Creating Confusion matrix and printing accuracy score.
from sklearn.metrics import confusion_matrix, accuracy_score                
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#Applying k-Fold Cross Validation to improve performance.
from sklearn.model_selection import cross_val_score                     
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
