import pandas as pd
from neural_network import *

# 1. Wczytanie i czyszczenie danych
df = pd.read_csv("housing.csv")
df = df.dropna()

if "ocean_proximity" in df.columns:
  df = df.drop("ocean_proximity", axis=1)

cel_kolumna = "median_house_value"
walidacja_danych(df, [cel_kolumna])

# 2. Oddzielamy cechy (X) od ceny docelowej (y)
X = df.drop(cel_kolumna, axis=1)
y = df[cel_kolumna]

# 3. Podział na zbiór treningowy i testowy (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test=0.2, ziarno=42)

# 4. Normalizacja za pomocą Twojej funkcji minmax (Złota zasada ML)
X_train_norm = minmax(X_train)
y_train_norm = minmax(y_train)

X_test_norm = minmax(X_test, min_val=X_train.min(), max_val=X_train.max())
y_test_norm = minmax(y_test, min_val=y_train.min(), max_val=y_train.max())

# Konwersja na listy do pętli sieci
dane_wejsciowe = X_train_norm.values.tolist()
ceny = y_train_norm.values.tolist()

dane_testowe = X_test_norm.values.tolist()
ceny_testowe = y_test_norm.values.tolist()

# 5. Inicjalizacja wag i biasów
wagi1 = stworz_wagi(dane_wejsciowe, 4)
biasy1 = stworz_biasy(4)

wagi2 = stworz_wagi([[0] * 4], 2)
biasy2 = stworz_biasy(2)

wagi3 = stworz_wagi([[0, 0]], 1)
biasy3 = stworz_biasy(1)

# Parametry treningowe
wspolczynnik_uczenia = 0.4
liczba_epok = 500

lr = wspolczynnik_uczenia
lr_zmniejszanie = 0.9
zmniejszanie_co = 20

print("--- START TRENINGU (ZBIÓR TRENINGOWY - LEAKY RELU) ---")

for epoka in range(liczba_epok):
  wyniki_koncowe = []

  # Wykładniczy Decay co 20 epok
  if epoka > 0 and epoka % zmniejszanie_co == 0:
    wspolczynnik_uczenia *= lr_zmniejszanie

  for i in range(len(dane_wejsciowe)):
    wiersz = dane_wejsciowe[i]
    prawdziwa_cena = ceny[i]

    # --- A. FORWARD PASS (Używamy leaky_relu!) ---
    z1, a1 = warstwa(wiersz, wagi1, biasy1, leaky_relu)
    z2, a2 = warstwa(a1, wagi2, biasy2, leaky_relu)
    z3, a3 = warstwa(a2, wagi3, biasy3, leaky_relu)

    wyniki_koncowe.append(a3)

    # --- B. BACKPROPAGATION (Używamy pochodna_leaky_relu!) ---
    blad3 = [a3[0] - prawdziwa_cena]
    blad2 = cofaj_blad(blad3, wagi3, z2, pochodna_leaky_relu)
    blad1 = cofaj_blad(blad2, wagi2, z1, pochodna_leaky_relu)

    # --- C. AKTUALIZACJA ---
    wagi3 = aktualizuj_wagi(a2, wagi3, blad3, wspolczynnik_uczenia)
    biasy3 = aktualizuj_biasy(biasy3, blad3, wspolczynnik_uczenia)

    wagi2 = aktualizuj_wagi(a1, wagi2, blad2, wspolczynnik_uczenia)
    biasy2 = aktualizuj_biasy(biasy2, blad2, wspolczynnik_uczenia)

    wagi1 = aktualizuj_wagi(wiersz, wagi1, blad1, wspolczynnik_uczenia)
    biasy1 = aktualizuj_biasy(biasy1, blad1, wspolczynnik_uczenia)

  aktualny_mse = mse(wyniki_koncowe, ceny)
  if epoka % 20 == 0:
    print(f"Epoka {epoka} | Błąd MSE (Train): {aktualny_mse:.6f}")

print("\n--- EWALUACJA NA ZBIORZE TESTOWYM ---")
wyniki_testowe = []
for wiersz in dane_testowe:
  _, a1 = warstwa(wiersz, wagi1, biasy1, leaky_relu)
  _, a2 = warstwa(a1, wagi2, biasy2, leaky_relu)
  _, a3 = warstwa(a2, wagi3, biasy3, leaky_relu)
  wyniki_testowe.append(a3[0])

test_mse = mse([[val] for val in wyniki_testowe], ceny_testowe)
print(f"Końcowy błąd MSE na danych testowych: {test_mse:.6f}")

print("\nPrzykładowe porównanie prognoz na danych testowych:")
for i in range(min(10, len(wyniki_testowe))):
  prognoza_usd = odwroc_minmax(
      wyniki_testowe[i], y_train.min(), y_train.max()
  )
  prawdziwa_usd = odwroc_minmax(ceny_testowe[i], y_train.min(), y_train.max())
  print(
      f"Przewidziana: ${prognoza_usd:,.2f} | Prawdziwa: ${prawdziwa_usd:,.2f}"
  )