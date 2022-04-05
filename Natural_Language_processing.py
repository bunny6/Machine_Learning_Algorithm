#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                        #importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3) #importing the tsv file. Delimiter is telling that we are import a tsv file and quoting=3 means ignore ''.


# In[3]:


import re                                 #import re=regular expression
import nltk                               #import natural language tool kit
nltk.download('stopwords')                #downloading the stopwords from nltk
from nltk.corpus import stopwords         #importing the stopwords from corpus module
from nltk.stem.porter import PorterStemmer   #for stemming, we import PorterStemmer
corpus=[]                                 #corpus means collection of documents.
for i in range(0,1000):                   #applying for loop for each comment should be stem
    review=re.sub('[^a-zA-Z]'," ",df["Review"][i])  #re.sub will replace a string by whatever we want.
    review=review.lower()                  #making all the alphabets lower case.
    review=review.split()                  #spliting means in short tokenization.['hey','this','example']
    ps=PorterStemmer()                     #Calling porterstemmer class and creating a object for it.
    all_stopwords=stopwords.words('english') #creating object for stopwords
    all_stopwords.remove('not')              #removing 'not' from stopwords.
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)]  #here we do stemming, removing the stopwords.
    review=" ".join(review)                 #now we join our words.
    corpus.append(review)                  #appending the words in corpus list which we have created.


# In[4]:


print(corpus)                  #printing the corpus


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer    #to convert the textual data into numerical format, we use Vectorization.
cv=CountVectorizer(max_features=1500)                          #to get rid of unnecessary words, we add max features.
X=cv.fit_transform(corpus).toarray()                           #fit will take al the words and transform will put all the words into matrix.Toaraay is used because naive bayes requires an array as input.
y=df.iloc[:,-1].values                                         #fetching the last column.


# In[6]:


from sklearn.model_selection import train_test_split            #split the data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[7]:


from sklearn.naive_bayes import GaussianNB  #naive bayes is one of the best algo for text analysis. Thats why we used here.
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[8]:


y_pred = classifier.predict(X_test)  #predicting the results and printing the results side by side.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[9]:


from sklearn.metrics import confusion_matrix, accuracy_score   #importing the accuracy metrics.
cm = confusion_matrix(y_test, y_pred)                          #printing the confusion matrix.
print(cm)
accuracy_score(y_test, y_pred)


# In[10]:


from sklearn.metrics import accuracy_score    #getting the performace of the model.
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[12]:


acc=accuracy_score(y_test,classifier.predict(X_test))
recall=recall_score(y_test,classifier.predict(X_test))
pre=precision_score(y_test,classifier.predict(X_test))
f1=f1_score(y_test,classifier.predict(X_test))
print(acc)
print(recall)
print(pre)
print(f1)


# In[ ]:




