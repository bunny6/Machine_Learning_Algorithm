#importing the necessary librarie

import numpy as np                       s
import pandas as pd
import matplotlib.pyplot as plt

 #importing the tsv file. Delimiter is telling that we are import a tsv file and quoting=3 means ignore ''.
df=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)


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




#printing the corpus
print(corpus)                  



from sklearn.feature_extraction.text import CountVectorizer    #to convert the textual data into numerical format, we use Vectorization.
cv=CountVectorizer(max_features=1500)                          #to get rid of unnecessary words, we add max features.
X=cv.fit_transform(corpus).toarray()                           #fit will take al the words and transform will put all the words into matrix.Toaraay is used because naive bayes requires an array as input.
y=df.iloc[:,-1].values                                         #fetching the last column.



 #split the data into training and testing.
    
from sklearn.model_selection import train_test_split           
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#naive bayes is one of the best algo for text analysis. Thats why we used here.

from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicting the results and printing the results side by side.

y_pred = classifier.predict(X_test)  
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


##importing the accuracy metrics and printing the confusion matrix and the accuracy.
from sklearn.metrics import confusion_matrix, accuracy_score   
cm = confusion_matrix(y_test, y_pred)                          
print(cm)
accuracy_score(y_test, y_pred)

#getting the performace of the model.

from sklearn.metrics import accuracy_score    
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


acc=accuracy_score(y_test,classifier.predict(X_test))
recall=recall_score(y_test,classifier.predict(X_test))
pre=precision_score(y_test,classifier.predict(X_test))
f1=f1_score(y_test,classifier.predict(X_test))
print(acc)
print(recall)
print(pre)
print(f1)







