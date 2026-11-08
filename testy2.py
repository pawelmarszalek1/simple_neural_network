import pandas as pd
from neural_network import *  # Dopasuj nazwę, jeśli Twój plik nazywa się inaczej

# 1. Wczytanie danych z pliku CSV
# (zmień 'housing.csv' na dokładną nazwę swojego pobranego pliku)
df = pd.read_csv('housing.csv')

# Najpierw usuwamy wiersze z brakującymi danymi (np. NaN w total_bedrooms)
df = df.dropna()

# Usuwamy kolumnę tekstową 'ocean_proximity', ponieważ sieć operuje wyłącznie na liczbach
if 'ocean_proximity' in df.columns:
  df = df.drop('ocean_proximity', axis=1)

# Ustawiamy kolumnę docelową (cena / mediana wartości domu w okręgu)
cel_kolumna = 'median_house_value'

# Walidacja danych za pomocą Twojej autorskiej funkcji
walidacja_danych(df, [cel_kolumna])

# Zapamiętujemy min i max dla ceny do późniejszego odwrócenia normalizacji
cena_min = df[cel_kolumna].min()
cena_max = df[cel_kolumna].max()

# 2. Normalizacja Min-Max całego DataFrame
df_znormalizowany = minmax(df)

# Rozdzielamy na dane wejściowe (X) i ceny docelowe (y)
dane_wejsciowe = df_znormalizowany.drop(cel_kolumna, axis=1).values.tolist()
ceny = df_znormalizowany[cel_kolumna].values.tolist()

# 3. Inicjalizacja wag i biasów dla 3 warstw
# Wagi pierwszej warstwy automatycznie dostosują się do liczby kolumn cech w CSV
wagi1 = stworz_wagi(dane_wejsciowe, 4)  # 4 neurony w 1. warstwie ukrytej
biasy1 = stworz_biasy(4)

wagi2 = stworz_wagi(
    [[0] * 4], 2
)  # 2 neurony w 2. warstwie ukrytej (wejście ma wymiar 4)
biasy2 = stworz_biasy(2)

wagi3 = stworz_wagi(
    [[0, 0]], 1
)  # 1 neuron na wyjście (wejście ma wymiar 2)
biasy3 = stworz_biasy(1)

# Parametry treningowe
wspolczynnik_uczenia = 0.4
liczba_epok = 500

lr = wspolczynnik_uczenia
lr_zmniejszanie = 0.9
zmniejszanie_co = 20

print(f'liczna epok: {liczba_epok}\nwspolczynnik uczenia: {wspolczynnik_uczenia}')
print('--- START TRENINGU NA CALIFORNIA HOUSING ---')

# Pętla epok (uczenia)
for epoka in range(liczba_epok+1):
  wyniki_koncowe = []

  for i in range(len(dane_wejsciowe)):
    wiersz = dane_wejsciowe[i]
    prawdziwa_cena = ceny[i]

    # --- A. FORWARD PASS (Zbieramy z oraz a) ---
    z1, a1 = warstwa(wiersz, wagi1, biasy1, leaky_relu)
    z2, a2 = warstwa(a1, wagi2, biasy2, leaky_relu)
    z3, a3 = warstwa(a2, wagi3, biasy3, leaky_relu)

    wyniki_koncowe.append(a3)

    # --- B. BACKPROPAGATION ---
    # Błąd dla wyjścia (różnica między prognozą a prawdą)
    blad3 = [a3[0] - prawdziwa_cena]

    # Cofamy błąd przez warstwy, podając uniwersalną funkcję pochodną
    blad2 = cofaj_blad(blad3, wagi3, z2, pochodna_leaky_relu)
    blad1 = cofaj_blad(blad2, wagi2, z1, pochodna_leaky_relu)

    # --- C. AKTUALIZACJA WAG I BIASÓW ---
    wagi3 = aktualizuj_wagi(a2, wagi3, blad3, wspolczynnik_uczenia)
    biasy3 = aktualizuj_biasy(biasy3, blad3, wspolczynnik_uczenia)

    wagi2 = aktualizuj_wagi(a1, wagi2, blad2, wspolczynnik_uczenia)
    biasy2 = aktualizuj_biasy(biasy2, blad2, wspolczynnik_uczenia)

    wagi1 = aktualizuj_wagi(wiersz, wagi1, blad1, wspolczynnik_uczenia)
    biasy1 = aktualizuj_biasy(biasy1, blad1, wspolczynnik_uczenia)

  # Sprawdzenie błędu MSE dla znormalizowanych wartości w danej epoce
  aktualny_mse = mse(wyniki_koncowe, ceny)

  if epoka % 5 == 0:
    print(f'Epoka {epoka} | Błąd MSE (znormalizowany): {aktualny_mse:.6f}')
  if epoka % zmniejszanie_co == 0:
        lr = lr * lr_zmniejszanie
        print(f"Epoka {epoka}: Zmniejszono LR do {lr:.5f}")

print('\n--- KONIEC TRENINGU ---')
print('Przykładowe porównanie prognoz z rzeczywistością (pierwsze 5 domów):')
for i in range(min(5, len(wyniki_koncowe))):
  prognoza_norm = wyniki_koncowe[i][0]
  prawdziwa_norm = ceny[i]

  # Odwracamy normalizację za pomocą Twojej funkcji, aby wrócić do oryginalnych cen w USD
  prognoza_usd = odwroc_minmax(prognoza_norm, cena_min, cena_max)
  prawdziwa_usd = odwroc_minmax(prawdziwa_norm, cena_min, cena_max)

  print(
      f'Przewidziana: ${prognoza_usd:,.2f} | Prawdziwa: ${prawdziwa_usd:,.2f}'
  )