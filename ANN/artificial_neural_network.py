# Importing the libraries


import numpy as np
import pandas as pd
import tensorflow as tf

#checking tensorflow version

tf.__version__

# Importing the dataset.

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)

print(y)

# Encoding categorical data by using label encoding on gender column.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)

#One Hot Encoding the "Geography" column.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

# Splitting the dataset into the Training set and Test set.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Applying Feature Scaling.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initializing the ANN

ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer.

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer.

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer.

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the ANN.

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set.

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Making the predictions and evaluating the model.

#Predicting on Test set results.

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Checking the performance of the model by Creating Confusion matrix and printing  accuracy score.

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
