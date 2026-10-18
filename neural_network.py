import random
import pandas as pd 
import math

df = pd.read_csv('test1.csv')
price = df['cena']
df.drop('cena',axis=1)
print(df,price)
def relu(x):
    return max(0,x)
def sigmoid(x):
    return 1/(1+math.e**-x)

def neuron(wejscie,wagi,bias,funkcja):
    if isinstance(wejscie,(list,pd.Series,pd.DataFrame)):
        wyjscie = sum(w * e for w, e in zip(wagi, wejscie)) + bias
    if isinstance(wejscie, (int,float)):
        wyjscie = (wejscie * wagi)+bias
    return funkcja(wyjscie)


def warstwa_wejsciowa(df):
    for i in range(len(df)):
        print(a)






