import pandas as pd
import math


a=[[1],[2],[3]]
b=[[4],[5],[6]]

df = pd.read_csv('test1.csv')
dane_wejsciowe= df.drop("cena", axis=1).values.tolist()
ceny = df['cena']

#print(ceny.values.tolist())


lista_prawdziwe = []
lista_bledow = []
czy_lista_list =( isinstance(ceny,list) and not isinstance(ceny[0],list))

if not czy_lista_list:
       lista_prawdziwe = [[x] for x in ceny]
       print("weszło")
else:
    lista_prawdziwe = ceny
    print('nie weszło')
lista_bledow = [[(x-y)**2 for x,y in zip(prog,wart)] for prog,wart in zip(dane_wejsciowe,lista_prawdziwe)]

print(lista_bledow)