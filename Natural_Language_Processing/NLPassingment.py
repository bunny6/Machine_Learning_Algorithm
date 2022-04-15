#!/usr/bin/env python
# coding: utf-8

# # importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd


# In[2]:


from nltk.corpus import PlaintextCorpusReader
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('wordnet')
nltk.download('punkt')
import string
import matplotlib.pyplot as plt


# # business corpus

# In[4]:



corpus_root1 = '/home/neosoft/Downloads/NLP/document_classification/bbc_fulltext_document_classification/bbc/business'

#business folder path


# In[5]:


filelists = PlaintextCorpusReader(corpus_root1, '.*')
#read all the text files in business folder


# In[6]:


a=filelists.fileids()


# In[7]:


wordslist = filelists.words('510.txt')


# In[8]:


print(wordslist)


# In[9]:


print(a)


# In[10]:


businessCorpus=[]


# In[11]:


for file in a:
  wordslist = filelists.words(file) #Read all the words in the each text file iterating through the loop
  businessCorpus.append(wordslist)


# In[12]:


print(businessCorpus)


# In[14]:


Bcorpus=[]
for item in businessCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower()  #Converting  words to lower case 
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2)
        if item2 not in set(stopwords.words('english')) and len(item2)>2: 
            #separating the words that are not stopwords and length of the words greater than 2
            new.append(item2)
    Bcorpus.append(new)   


# In[15]:


print(Bcorpus) 
#business corpus array after removing stopwords and cnverting to lower case and applying lemmatization


# In[161]:


Bcorpus1=[]
for i in Bcorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)
      
    Bcorpus1.append(new)    
      


# In[162]:


print(Bcorpus1)
#Business array after removing the empty values


# In[18]:


print(Bcorpus1[0])


# In[20]:


Bcorpus2=[]
for i in Bcorpus1:
    Bcorpus2.append(" ".join(i))


# In[21]:


print(Bcorpus2) 
#Business list after converting the words after doing limatization and finding unique words into string in each document


# In[22]:


df1=pd.DataFrame({'page':Bcorpus2,'class':"Business"})
#Business Class DataFrame


# In[23]:


df1["Text"]=Bcorpus1
print(df1)
#added new column in the business Dataframe which contains the list of Bag of words created


# # Entertainment

# In[24]:



corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/entertainment'
#path of Entertainment Folder


# In[25]:


filelists = PlaintextCorpusReader(corpus_root1, '.*')


# In[26]:


a=filelists.fileids()
#list containing all the text files from Entertainment Folder


# In[27]:


print(a)


# In[28]:


entertainmentCorpus=[]


# In[29]:


for file in a:
    wordslist = filelists.words(file)
    #Read all the words in each file of Entertainment Folder
    entertainmentCorpus.append(wordslist)


# In[30]:


print(entertainmentCorpus)


# In[31]:


print(entertainmentCorpus)


# In[163]:


Ecorpus=[]
for item in entertainmentCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower()  #converted each word to lower case 
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #applied lemmatization on each word
        if item2 not in set(stopwords.words('english')) and len(item2)>2:
            new.append(item2)
    Ecorpus.append(new)   


# In[164]:


print(Ecorpus) #Entertainment Array after applying lemmatization,changing tom lower case,replacing punctuations.


# In[165]:


Ecorpus1=[]
for i in Ecorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)

    Ecorpus1.append(new)    


# In[166]:


print(Ecorpus1) #Entertainment Array after removing null elements and unique words


# In[36]:


print(Ecorpus1[0])


# In[37]:


Ecorpus2=[]
for i in Ecorpus1:
    Ecorpus2.append(" ".join(i))


# In[38]:


print(Ecorpus2) # Entertainment Array after making all words in each text file to string.


# In[167]:


df2=pd.DataFrame({'page':Ecorpus2,'class':"Entertainment"})
#Entertainment Data Frame


# In[168]:


df2["Text"]=Ecorpus1
#Added new column which has the Bagwords as rows
print(df2)


# # politics

# In[41]:



corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/politics'
#Politics Folder Path


# In[42]:


filelists = PlaintextCorpusReader(corpus_root1, '.*')


# In[43]:


a=filelists.fileids()
#Read all the text files in the entertainment Folder


# In[44]:


print(a)


# In[45]:


politicsCorpus=[]


# In[46]:


for file in a:
    wordslist = filelists.words(file)
    #Read all the words in each text file of Politics folder text files
    politicsCorpus.append(wordslist)


# In[48]:


print(politicsCorpus)


# In[169]:


Pcorpus=[]
for item in politicsCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower() #changing case of the letter to lower case
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #Applied Lemmatization on each word of Politics text files
        if item2 not in set(stopwords.words('english')) and len(item2)>2: #Words which are not in stopwords and words length greater than 2 are found
            new.append(item2)
    Pcorpus.append(new)   


# In[50]:


print(Pcorpus) #Politics Array after applying Lemmatization and removing stopwords


# In[171]:


Pcorpus1=[]
for i in Pcorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)
      
    Pcorpus1.append(new)    
      


# In[52]:


print(Pcorpus1) #Politics Array after removing empty elements and finding the unique words


# In[53]:


print(Pcorpus1[0])


# In[54]:


Pcorpus2=[]
for i in Pcorpus1:
    Pcorpus2.append(" ".join(i))


# In[55]:


print(Pcorpus2)#Politics Array after joining the words in each text file to form a string


# In[56]:


df3=pd.DataFrame({'page':Pcorpus2,'class':"Politics"})
#Data frame for Politics


# In[57]:


df3["Text"]=Pcorpus1
#added new column to Politics DataFrame which has bag of words of each text file
print(df3)


# # Sport

# In[58]:



corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/sport'
#sport folder Path


# In[59]:


filelists = PlaintextCorpusReader(corpus_root1, '.*')


# In[172]:


a=filelists.fileids()
#list Containg all the text files of Sport Folder


# In[173]:


print(a)


# In[62]:


sportsCorpus=[]


# In[63]:


import codecs
new=[]
for file in a:
    f = open('document_classification/bbc_fulltext_document_classification/bbc/sport/{}'.format(file), 'r', encoding="latin-1")
    #Got an utf-8 error so used encoding while reading the text in the file
    text_data=f.read().split('\n')
  
    text_data = list(filter(None, text_data))
    
    new.append(text_data)
    
print(new)


# In[174]:


new1=[]
for i in new:
    a=' '.join(i)
   
    new1.append(a)
    
  


# In[65]:


print(new1)
#Joined each word in the file to form a string in Sports Array


# In[66]:



for i in new1:
    sportsCorpus.append(i.split())
    


# In[67]:


print(sportsCorpus)# done toneization for each file in Sports Array


# In[175]:


Scorpus=[]
for item in sportsCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower()  #converted to lower case
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #Applying Lemmatization on  Each word
        if item2 not in set(stopwords.words('english')) and len(item2)>2:
            new.append(item2)
    Scorpus.append(new)   


# In[69]:


print(Scorpus) #sports Array after removing stopwords ,converting to lower case words


# In[70]:


Scorpus1=[]
for i in Scorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)
      
    Scorpus1.append(new)    
       
      
        


# In[71]:


print(Scorpus1) #sports array after removing the empty elements and finding the unique words


# In[72]:


Scorpus2=[]
for i in Scorpus1:
    Scorpus2.append(" ".join(i))


# In[73]:


print(Scorpus2) #sports array after making string of words from each file 


# In[74]:


df4=pd.DataFrame({'page':Scorpus2,'class':"Sport"})
#data frame for sports class


# In[75]:



df4["Text"]=Scorpus1
#added new column containing the list of words of each text file
print(df4)


# # Tech

# In[76]:



corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/tech'
#Tech Folder path


# In[77]:


filelists = PlaintextCorpusReader(corpus_root1, '.*')


# In[78]:


a=filelists.fileids()
#list containing all the files of tech folder


# In[79]:


print(a)


# In[80]:


techCorpus=[]


# In[81]:


for file in a:
    wordslist = filelists.words(file) #read all the words from each file of tech folder
    techCorpus.append(wordslist)


# In[82]:


print(techCorpus)


# In[83]:


print(techCorpus)


# In[176]:


Tcorpus=[]
for item in techCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower() #converted to lower case  
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #applied lemmatization 
        if item2 not in set(stopwords.words('english')) and len(item2)>2: #removed stopwords and words less than size 3
            new.append(item2)
    Tcorpus.append(new)   


# In[177]:


print(Tcorpus) #Tech array after removing stopwords and doing lemmatization 


# In[86]:


Tcorpus1=[]
for i in Tcorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)

    Tcorpus1.append(new)    
      


# In[87]:


print(Tcorpus1) #Tech array after removing empty elements and duplicates


# In[88]:


print(Tcorpus1[0])


# In[89]:


Tcorpus2=[]
for i in Tcorpus1:
    Tcorpus2.append(" ".join(i))


# In[90]:


print(Tcorpus2) # tech array after joing the list elements of each file in tech folder


# In[91]:


df5=pd.DataFrame({'page':Tcorpus2,'class':"Tech"})
#Dataframe for Tech 
df5["Text"]=Tcorpus1
#added new column in Tech dataframe 


# In[92]:


print(df5)


# In[124]:


DF=pd.concat((df1,df2,df3,df4,df5))
#dataframe after concatenating all the dataframes tech,sport,entertainment,politics and business


# In[125]:


print(DF)


# In[126]:


DF = DF.rename(columns={'page': 'page', 'class': 'category'})
#renamed column in dataframe


# In[127]:


print(DF)


# # tfidf vectorizor

# In[128]:


#applied tfidf vectorizor
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(input='content', analyzer = 'word', lowercase=True, stop_words='english',                                   ngram_range=(1, 3), min_df=40, max_df=0.20,                                  norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
text_vector = vectorizer.fit_transform(DF.page)
dtm = text_vector.toarray()
features = vectorizer.get_feature_names()


# # Label Encoding

# In[129]:



from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
DF['label'] = label_enc.fit_transform(DF['category'])
DF.tail()


# In[130]:


DF[DF['label']==3]


# In[131]:


print(text_vector)


# In[132]:


h = pd.DataFrame(data = text_vector.todense(), columns = vectorizer.get_feature_names())
h.iloc[:,:]


# In[133]:


pip install -U gensim


# In[134]:


from gensim.corpora import Dictionary
# dictionary = Dictionary(DF['Text'])
# corpus = [dictionary.doc2bow(txt) for txt in DF["Text"]]


# In[135]:


dictionary = Dictionary(DF['Text'])
print('Nr. of unique words in initital documents:', len(dictionary))

# Filter out words that occur less than 10 documents, or more than 20% of the documents
dictionary.filter_extremes(no_below=10, no_above=0.2)
print('Nr. of unique words after removing rare and common words:', len(dictionary))


# In[159]:



print(dictionary)


# In[ ]:





# In[ ]:





# In[151]:


X = text_vector
y = DF.label.values


# In[152]:


DF[DF["label"]==0]


# # splitting the data into train and test

# In[153]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[154]:


print(X_train)
print(y_train)


# # Model Training

# ## Random Forest

# In[155]:


from sklearn.ensemble import RandomForestClassifier


# In[156]:


svc1 = RandomForestClassifier(random_state = 0)
svc1.fit(X_train, y_train)
svc1_pred = svc1.predict(X_test)
#print(f"Train Accuracy: {svc1.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc1.score(X_test, y_test)*100:.3f}%")


# # K Neighbours

# In[157]:


from sklearn.neighbors import KNeighborsClassifier


# In[158]:


svc4 = KNeighborsClassifier()
#pprint(svc4.get_params())
svc4.fit(X_train, y_train)
svc4_pred = svc4.predict(X_test)
#print(f"Train Accuracy: {svc4.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc4.score(X_test, y_test)*100:.3f}%")


# In[ ]:





# In[ ]:




