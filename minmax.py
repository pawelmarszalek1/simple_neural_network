from neural_network import *

df = pd.read_csv('test1.csv')

ceny = df['cena']
ceny_norm = minmax(ceny)
norm = minmax(df)
predykcje_modelu = pd.Series([0.020755562139530424, 0.020755541383968283, 0.020755554774768363, 0.49028758986358284, 0.045244328506606986, 0.49294171923180335])
ceny_org = odwroc_minmax(predykcje_modelu, ceny.min(),ceny.max())

print(ceny_norm)
print(ceny_org)