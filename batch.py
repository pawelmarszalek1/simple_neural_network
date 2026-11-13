from neural_network import *

X = [[1,2,2],[2,1,3],[3,1,2],[6,1,1]]
W = [1,2,4]
b=[2]
Y = [5,10,8,9]
a=[]
mse=0
n = 0.01


z=[sum((si*w) for si,w in zip(s,W))+b for s in X]
for l in range(len(z)):    
    a.append(max(0,z[l]))

for k in range(len(a)):
    mse += (a[k]-Y[k])**2

mse /= len(X)

delta = [2/len(X) * (pred-ocz) * pochodna_relu(z_i) for pred,ocz,z_i in zip(a,Y,z)]


pochodna_L_wzgedem_W = []
suma = 0
for wiersz in range(len(X[0])):
    for kolumna in range(len(X)):
        print(f'X= {X[kolumna][wiersz]}* {delta[kolumna]}')
        suma +=X[kolumna][wiersz] * delta[kolumna]
    print('wyjscie')
    pochodna_L_wzgedem_W.append(suma)
    suma=0

pochodna_L_wzgedem_b =sum(delta)

W_nowe = [w_stare - (n * wart_poch) for w_stare,wart_poch in zip(W,pochodna_L_wzgedem_W)]
b_nowe = b - (n*pochodna_L_wzgedem_b)
'''
print(f"z= {z}")
print(f"a= {a}")
print(f"mse= {mse}")
print(f"delta= {delta}")
print(f"L wzgledem w= {pochodna_L_wzgedem_W}")
print(f"L wzgledem b= {pochodna_L_wzgedem_b}")
print(f'nowe wagi= {W_nowe}')
print(f'nowy bias= {b_nowe}')
'''

w1 = warstwa_batch(X,W,b,pochodna_relu)

print(w1)