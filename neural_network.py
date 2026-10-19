import random
import pandas as pd 
import math
from functools import reduce

#-------------|
# FORWARD PASS|
#-------------|
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

def stworz_wagi(dane_wejscia,liczba_neuronow):
    liczba_wag = len(dane_wejscia[0])
    wagi = [[random.uniform(-1,1) for _ in range (liczba_wag)] for _ in range(liczba_neuronow)]
    return wagi

def stworz_biasy(liczba_neuronow):
    biasy = [random.uniform(-0.1, 0.1) for _ in range(liczba_neuronow)]
    return biasy

#-------------|
# FORWARD PASS|
#-------------|

#---------------|
# Funkcje straty|
#---------------|
def mse(prognozy,prawdziwe_wartosci):
    lista_prawdziwe = []
    lista_bledow = []
    czy_lista_list = ( isinstance(prawdziwe_wartosci,list) and isinstance(prawdziwe_wartosci[0],list))
    if not czy_lista_list:
        lista_prawdziwe = [[x] for x in prawdziwe_wartosci] # zmiana z listy na liste liist zeby sie zgadzalo z tym co jest na wejsicu
    else:
        lista_prawdziwe = prawdziwe_wartosci
    lista_bledow = [[(x-y)**2 for x,y in zip(prog,wart)] for prog,wart in zip(prognozy,lista_prawdziwe)]
    suma = sum([element for podlista in lista_bledow for element in podlista]) # zmiana z listy list zeby uzyc sum
    calkowita_liczba_wartosci = sum(len(prog) for prog in prognozy)
    return suma/calkowita_liczba_wartosci
