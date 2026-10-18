import random
import pandas as pd 
import math

df = pd.read_csv('test1.csv')
price = df['cena']
df=df.drop('cena',axis=1)
#print(df)
def relu(x):
    return max(0,x)
def sigmoid(x):
    return 1/(1+math.e**-x)

def neuron(wejscie,wagi,bias,funkcja_aktywacji):
    if isinstance(wejscie,(list ,pd.Series,pd.DataFrame)):
        wyjscie = sum(w * e for w, e in zip(wagi, wejscie)) + bias
    if isinstance(wejscie, (int,float)):
        wyjscie = (wejscie * wagi)+bias
    return funkcja_aktywacji(wyjscie)


def warstwa(wejscie, wagi_warstwy, bias, funkcja_akt):
    lista_wyjscie = [neuron(wejscie, w, b, funkcja_akt) for w, b in zip(wagi_warstwy, bias)]
    return lista_wyjscie


wagi = [[5,2,3,2],[3,2,2,4]]
biasy = [3,4]

df = pd.read_csv('test1.csv')
dane= df.drop("cena", axis=1).values.tolist()
print(warstwa(dane[0],wagi,biasy,relu))

#x=warstwa(dane[0],wagi,biasy[0],relu)

#print(warstwa(x,wagi,biasy,relu))





