from neural_network import *


df = pd.read_csv('test1.csv')
dane_wejsciowe= df.drop("cena", axis=1).values.tolist()
ceny = df['cena']

wyniki = []

w = [0.1,0.2,0.001,0.1]
w1 = stworz_wagi(dane_wejsciowe,1)

print('wagi ',w1)

print(neuron(dane_wejsciowe[0],w,0,relu))

for wiersze in dane_wejsciowe:
    z,a=warstwa(wiersze,w1,[1],leaky_relu)
    wyniki.append(a)
    print(f'Dane: {wiersze}, wagi: {w1}, wynik: {a}')


funkcja_straty = mse(wyniki,ceny)

print(funkcja_straty)

