import random
import pandas as pd 
import math


# ==============
# napisać funkcje walidacji danych i wag moge trzymac oczekiwana ilosc wag w jakiejs zminnej otrzeba
#  rozważyć przypadki dla danych x(danych wejsciowych) i y(etykiet)
# ==============

def walidacja_danych(dane,etykiety):
    if dane.empty:
        raise ValueError("Przesłany plik jest pusty")
    brakujace = [kol for kol in etykiety if kol not in dane.columns]
    if brakujace:
        raise KeyError(
            f"Brakujące kolumny w danych: {brakujace}. "
            f"Dostępne kolumny: {list(dane.columns)}"
        )
    if dane.isna().any().any():
        raise ValueError("Plik posiada brakujące dane")
    


#-------------|
# FORWARD PASS|
#-------------|
def relu(x):
    return max(0,x)
def sigmoid(x):
    return 1/(1+math.e**-x)
def leaky_relu(x, alpha=0.01):
    return max(alpha * x, x)

def neuron(wejscie, wagi, bias, funkcja_aktywacji):
  z = sum(w * e for w, e in zip(wagi, wejscie)) + bias
  a = funkcja_aktywacji(z)
  return z, a  # tu zmiana na zwracanie pary z - czysta suma, a aktywacja


def warstwa(wejscie, wagi_warstwy, bias, funkcja_akt):
  z_lista = []
  a_lista = []
  for w, b in zip(wagi_warstwy, bias):
    z, a = neuron(wejscie, w, b, funkcja_akt)
    z_lista.append(z)
    a_lista.append(a)
  return z_lista, a_lista # tu to samo co w neuronie zwraca pare listy 

def stworz_wagi(dane_wejscia,liczba_neuronow):
    liczba_wag = len(dane_wejscia[0])
    wagi = [[random.uniform(0.1,1) for _ in range (liczba_wag)] for _ in range(liczba_neuronow)]
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

#---------------|
# Funkcje straty|
#---------------|



def aktualizuj_wagi(wejscie, wagi, blad, wspolczynnik_uczenia):
    for i in range(len(wagi)):
        for j in range(len(wagi[i])):
            gradient_wagi = blad[i] * wejscie[j]
            wagi[i][j] = wagi[i][j] - (wspolczynnik_uczenia * gradient_wagi)
    return wagi

def aktualizuj_biasy(biasy, blad, wspolczynnik_uczenia):
    for i in range(len(biasy)):
        biasy[i] = biasy[i] - (wspolczynnik_uczenia * blad[i])
    return biasy

def pochodna_relu(wartosc):
    return 1 if wartosc > 0 else 0

def pochodna_leaky_relu(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def cofaj_blad(blad_nastepny, wagi_nastepne, z_obecne,funkcja_pochodna):
  liczba_neuronow_obecnych = len(z_obecne)
  blad_obecny = []

  for j in range(liczba_neuronow_obecnych):
    suma = 0
    for i in range(len(blad_nastepny)):
      suma += blad_nastepny[i] * wagi_nastepne[i][j]

    pochodna = funkcja_pochodna(z_obecne[j])
    blad_obecny.append(suma * pochodna)

  return blad_obecny
        
def minmax(df):
    norm = (df-df.min())/(df.max()-df.min())
    return norm

def odwroc_minmax(wartosci_normalizowane,orginalne_min,orginalne_max):
    orinalne = wartosci_normalizowane * (orginalne_max-orginalne_min) + orginalne_min
    return orinalne