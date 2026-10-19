from neural_network import *


df = pd.read_csv('test1.csv')
dane_wejsciowe= df.drop("cena", axis=1).values.tolist()
ceny = df['cena']

dane_z_warstwy1 = []
dane_z_warstwy2 = []
dane_z_warstwy3 = []

wagi1 = stworz_wagi(dane_wejsciowe,3)
biasy1 = stworz_biasy(3)
for wiersze in dane_wejsciowe:
    jeden_dom = warstwa(wiersze,wagi1,biasy1,relu)
    dane_z_warstwy1.append(jeden_dom)

print("Wagi: ", wagi1)
print("Biasy: ", biasy1)
print("pierwsza warstwa ukryta: ",dane_z_warstwy1)

wagi2 = stworz_wagi(dane_z_warstwy1,2)
biasy2 = stworz_biasy(2)

for wiersze2 in dane_z_warstwy1:
    jeden_dom2 = warstwa(wiersze2,wagi2,biasy2,relu)
    dane_z_warstwy2.append(jeden_dom2)

print("\nWagi: ", wagi2)
print("Biasy: ", biasy2)
print("Druga warstwa ukryta: ",dane_z_warstwy2)

wagi3 = stworz_wagi(dane_z_warstwy2,1)
biasy3 = stworz_biasy(1)

for wiersze3 in dane_z_warstwy2:
    jeden_dom3 = warstwa(wiersze3,wagi3,biasy3,relu)
    dane_z_warstwy3.append(jeden_dom3)

print("\nWagi: ", wagi3)#print("Biasy: ", biasy3)
print("Wynik koncowy: ",dane_z_warstwy3)

bledy = mse(dane_z_warstwy3,ceny)
print("Błąd mse: ",bledy)