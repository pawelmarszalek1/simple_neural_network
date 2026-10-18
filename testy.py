import pandas as pd 

df = pd.read_csv('test1.csv')

def relu(x):
    return max(0,x)
def neuron(wejscie,wagi,bias,funkcja_akt):
    if isinstance(wejscie,(list ,pd.Series,pd.DataFrame)):
        wyjscie = sum(w * e for w, e in zip(wagi, wejscie)) + bias
    if isinstance(wejscie, (int,float)):
        wyjscie = (wejscie * wagi)+bias
    return funkcja_akt(wyjscie) 

df = pd.read_csv('test1.csv')
wejscie = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
wagi = [[5,2,3,2,4],[3,2,2,4,5]]
biasy = [3,4,5,6,7]


comp = [neuron(data,w,b,relu) for data,w,b in zip(wejscie,wagi,biasy)]


print(comp)