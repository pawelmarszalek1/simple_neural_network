import pandas as pd
from neural_network import *  # Dopasuj nazwę, jeśli Twój plik nazywa się inaczej

# 1. Wczytanie danych z pliku CSV
df = pd.read_csv('housing.csv')

# Najpierw usuwamy wiersze z brakującymi danymi (np. NaN w total_bedrooms)
df = df.dropna()

# Usuwamy kolumnę tekstową 'ocean_proximity'
if 'ocean_proximity' in df.columns:
  df = df.drop('ocean_proximity', axis=1)

# Ustawiamy kolumnę docelową
cel_kolumna = 'median_house_value'

# Walidacja danych
walidacja_danych(df, [cel_kolumna])

# Zapamiętujemy min i max dla ceny
cena_min = df[cel_kolumna].min()
cena_max = df[cel_kolumna].max()

# 2. Normalizacja Min-Max całego DataFrame
df_znormalizowany = minmax(df)

# Rozdzielamy na dane wejściowe (X) i ceny docelowe (y)
dane_wejsciowe = df_znormalizowany.drop(cel_kolumna, axis=1).values.tolist()
ceny = df_znormalizowany[cel_kolumna].values.tolist()

# 3. ROZBUDOWA ARCHITEKTURY (Więcej warstw i neuronów!)
# Zaczynamy od bezpieczniejszego startowego learning rate (np. 0.05 zamiast 0.4)
wspolczynnik_uczenia = 0.001
liczba_epok = 200
lr = wspolczynnik_uczenia
lr_zmniejszanie = 0.95
zmniejszanie_co = 20

# Definiujemy 4 warstwy ukryte/wyjściową
wagi1 = stworz_wagi(dane_wejsciowe, 16)  # 1. warstwa: 16 neuronów
biasy1 = stworz_biasy(16)

wagi2 = stworz_wagi([[0] * 16], 16)  # 2. warstwa: 16 neuronów
biasy2 = stworz_biasy(16)

wagi3 = stworz_wagi([[0] * 16], 8)  # 3. warstwa: 8 neuronów
biasy3 = stworz_biasy(8)

wagi4 = stworz_wagi([[0] * 8], 1)  # 4. warstwa (Wyjście): 1 neuron
biasy4 = stworz_biasy(1)

print(
    f'liczba epok: {liczba_epok}\nwspolczynnik uczenia (startowy):'
    f' {wspolczynnik_uczenia}'
)
print('--- START TRENINGU GŁĘBOKIEJ SIECI ---')

# Pętla epok (uczenia)
for epoka in range(liczba_epok + 1):
  wyniki_koncowe = []

  for i in range(len(dane_wejsciowe)):
    wiersz = dane_wejsciowe[i]
    prawdziwa_cena = ceny[i]

    # --- A. FORWARD PASS (4 warstwy) ---
    z1, a1 = warstwa(wiersz, wagi1, biasy1, leaky_relu)
    z2, a2 = warstwa(a1, wagi2, biasy2, leaky_relu)
    z3, a3 = warstwa(a2, wagi3, biasy3, leaky_relu)
    z4, a4 = warstwa(a3, wagi4, biasy4, leaky_relu)

    wyniki_koncowe.append(a4)

    # --- B. BACKPROPAGATION (Cofanie błędu przez 4 warstwy) ---
    blad4 = [a4[0] - prawdziwa_cena]

    blad3 = cofaj_blad(blad4, wagi4, z3, pochodna_leaky_relu)
    blad2 = cofaj_blad(blad3, wagi3, z2, pochodna_leaky_relu)
    blad1 = cofaj_blad(blad2, wagi2, z1, pochodna_leaky_relu)

    # --- C. AKTUALIZACJA WAG I BIASÓW (używamy aktywnej zmiennej 'lr'!) ---
    wagi4 = aktualizuj_wagi(a3, wagi4, blad4, lr)
    biasy4 = aktualizuj_biasy(biasy4, blad4, lr)

    wagi3 = aktualizuj_wagi(a2, wagi3, blad3, lr)
    biasy3 = aktualizuj_biasy(biasy3, blad3, lr)

    wagi2 = aktualizuj_wagi(a1, wagi2, blad2, lr)
    biasy2 = aktualizuj_biasy(biasy2, blad2, lr)

    wagi1 = aktualizuj_wagi(wiersz, wagi1, blad1, lr)
    biasy1 = aktualizuj_biasy(biasy1, blad1, lr)

  # Sprawdzenie błędu MSE
  aktualny_mse = mse(wyniki_koncowe, ceny)

  if epoka % 5 == 0:
    print(f'Epoka {epoka} | Błąd MSE (znormalizowany): {aktualny_mse:.6f}')

  if epoka > 0 and epoka % zmniejszanie_co == 0:
    lr = lr * lr_zmniejszanie
    print(f'Epoka {epoka}: Zmniejszono LR do {lr:.5f}')

print('\n--- KONIEC TRENINGU ---')
print('Przykładowe porównanie prognoz z rzeczywistością (pierwsze 5 domów):')
for i in range(min(5, len(wyniki_koncowe))):
  prognoza_norm = wyniki_koncowe[i][0]
  prawdziwa_norm = ceny[i]

  prognoza_usd = odwroc_minmax(prognoza_norm, cena_min, cena_max)
  prawdziwa_usd = odwroc_minmax(prawdziwa_norm, cena_min, cena_max)

  print(
      f'Przewidziana: ${prognoza_usd:,.2f} | Prawdziwa: ${prawdziwa_usd:,.2f}'
  )