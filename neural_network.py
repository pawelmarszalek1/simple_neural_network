import pandas as pd 
import random
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
    wyniki = [neuron(wejscie, w, b, funkcja_akt) for w, b in zip(wagi_warstwy, bias)]

    z_lista = [z for z, a in wyniki]
    a_lista = [a for z, a in wyniki]

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

def pochodna_mse(y_pred, y_praw, N):
    return (2 / N) * (y_pred - y_praw)

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
        
def minmax(df,min_val=None, max_val=None):
    if min_val is None:
        min_val = df.min()
    if max_val is None:
        max_val = df.max()
    norm = (df-min_val)/(max_val-min_val)
    return norm

def odwroc_minmax(wartosci_normalizowane,orginalne_min,orginalne_max):
    orinalne = wartosci_normalizowane * (orginalne_max-orginalne_min) + orginalne_min
    return orinalne


def train_test_split(dane,etykiety,test=0.2,ziarno=None):
    if ziarno is not None:
        random.seed(ziarno)

    l = len(dane)
    indeksy = list(range(l))
    random.shuffle(indeksy)

    podzial = int(l*(1-test))

    id_trening = indeksy[:podzial]
    id_test = indeksy[podzial:]

    if hasattr(dane, "iloc"):
        dane_trening = dane.iloc[id_trening]
        dane_test = dane.iloc[id_test]
    else:
        dane_trening = [dane[i] for i in id_trening]
        dane_test = [dane[i] for i in id_test]

    if hasattr(etykiety, "iloc"):
        etykiety_trening = etykiety.iloc[id_trening]
        etykiety_test = etykiety.iloc[id_test]
    else:
        etykiety_trening = [etykiety[i] for i in id_trening]
        etykiety_test = [etykiety[i] for i in id_test]

    return dane_trening, dane_test, etykiety_trening, etykiety_test


#=====================
#przetwarzanie wsadowe
#=====================

def warstwa_batch(batch_wejscie,wagi_warstwy,biasy,funkcja_aktywacji):
    batch_z = []
    batch_a = []
    for probka in batch_wejscie:
        z_lista,a_lista = warstwa(probka,wagi_warstwy,biasy,funkcja_aktywacji)
        batch_z.append(z_lista)
        batch_a.append(a_lista)
    return batch_z,batch_a

def delta_batch_wyjscia(batch_z_wyjscie,batch_a_wyjscie,batch_y,batch_rozmiar, pochodna_akt,pochodna_loss):
    delta = []
    for a_probka,y_probka,z_probka in zip(batch_a_wyjscie,batch_y,batch_z_wyjscie):
        delta_probki =[]
        for a_n,y_n,z_n in zip (a_probka,y_probka,z_probka):
            d_L = pochodna_loss(batch_rozmiar,a_n,y_n)
            d_akt = pochodna_akt(z_n)
            delta_probki.append(d_L*d_akt)
        delta.append(delta_probki)

    return delta

def cofaj_blad_batch(batch_delta_nastepna,wagi_nastepne,batch_z_obecne,funckja_pochodna):
    batch_delta_obecna = []
    for delta_nastepna, z_obecne in zip(batch_delta_nastepna,batch_z_obecne):
        d=cofaj_blad(delta_nastepna,wagi_nastepne,z_obecne,funckja_pochodna)
        batch_delta_obecna.append(d)
    return batch_delta_obecna


#aktualizuje oba naraz bo bez sensu robić jeszcze raz dwie takie same petle 
def aktualizuj_wagi_batch(batch_wejscie,batch_delty,wagi,biasy,wspolczynnik_uczenia):
    liczba_neuronow = len(wagi)
    liczba_wejsc = len(wagi[0])
    grad_wag = [[0.0 for _ in range(liczba_wejsc)] for _ in range(liczba_neuronow)]
    grad_bias = [0.0 for _ in range(liczba_neuronow)]

    for probka_x, probka_del in zip(batch_wejscie,batch_delty):
        for i in range(liczba_neuronow):
            for j in range(liczba_wejsc):
                grad_wag[i][j]+= probka_x[j]*probka_del[i]
            grad_bias+=probka_del[i]

    for i in range(liczba_neuronow):
        for j in range(liczba_wejsc):
          wagi[i][j] -= wspolczynnik_uczenia*grad_wag[i][j]
        biasy[i] -= wspolczynnik_uczenia*grad_bias[i]

    return wagi,biasy

def aktualizuj_wagi_i_biasy_batch(batch_wejscie, batch_delta, wagi, biasy, wspolczynnik_uczenia):
    liczba_neuronow = len(wagi)
    liczba_wejsc = len(wagi[0])
    rozmiar_batch = len(batch_wejscie)

    # 1. Inicjalizacja akumulatorów gradientów zerami
    grad_wagi = [[0.0 for _ in range(liczba_wejsc)] for _ in range(liczba_neuronow)]
    grad_biasy = [0.0 for _ in range(liczba_neuronow)]

    # 2. Akumulacja gradientów z każdej próbki w batchu
    for probka_x, probka_delta in zip(batch_wejscie, batch_delta):
        for i in range(liczba_neuronow):
            for j in range(liczba_wejsc):
                grad_wagi[i][j] += probka_delta[i] * probka_x[j]
            grad_biasy[i] += probka_delta[i]

    # 3. Aktualizacja parametrów
    for i in range(liczba_neuronow):
        for j in range(liczba_wejsc):
            sredni_grad_wagi = grad_wagi[i][j]/rozmiar_batch #usrednianie wag
            wagi[i][j] -= wspolczynnik_uczenia * sredni_grad_wagi[i][j]

        sredni_grad_biasy = grad_biasy[i] / rozmiar_batch #uśredninianie biasu
        biasy[i] -= wspolczynnik_uczenia * sredni_grad_biasy[i]

    return wagi, biasy