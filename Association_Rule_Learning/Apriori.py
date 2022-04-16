#importing the library.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#installing apriori
get_ipython().system('pip install apyori')

#Importing the datset.
df=pd.read_csv("Market_Basket_Optimisation.csv", header= None)

df.head()

df.describe()  #describe will return us all mean,median,min,max,std_dev for all the numerical columns.

df.info()  #info will return is the column name and the dtype. Helpful when dealing with garbage values.

df.isnull().sum()  #checking for null values

df.shape  #shape will return us the number of columns and rows of our df.

transactions=[]
for i in range(0,7501):
    transactions.append([str(df.values[i,j])for j in range(0,20)])

#importing apriori.   
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


result=list(rules)

print(result)

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in result]
    rhs         = [tuple(result[2][0][1])[0] for result in result]
    supports    = [result[1] for result in result]
    confidences = [result[2][0][2] for result in result]
    lifts       = [result[2][0][3] for result in result]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(result), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print(resultsinDataFrame)

resultsinDataFrame.nlargest(n = 10, columns = 'Lift')





