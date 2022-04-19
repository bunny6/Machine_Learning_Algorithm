#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading Dataset
df=pd.read_csv("Datasets/Ads_CTR_Optimisation.csv")
print(df)

#Implementing Thompsons sampling
import random
N=500
d=10
ads_selected=[]
number_of_rewards_1=[0] * d
number_of_rewards_0=[0] * d
total_rewards=0

for n in range(N):
    ad=0
    max_random=0
    for i in range(d):
        random_beta=random.betavariate(number_of_rewards_1[i] + 1 ,number_of_rewards_0[i] +1)
        print(random_beta)
        if (random_beta>max_random):
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=df.values[n,ad]
    if reward==1:
        number_of_rewards_1[ad]=number_of_rewards_1[ad] + 1
    else:number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1   
    total_rewards=total_rewards + reward
print(ads_selected)
print(total_rewards)

#Visualizing the results
plt.hist(ads_selected)
plt.title("Thompson's Sampling")
plt.xlabel("Ads")
plt.ylabel("Number of times ad Selected")
plt.show()


