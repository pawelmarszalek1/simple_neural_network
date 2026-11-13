import pandas as pd
import math

gradienty = [
    [1, -2],
    [5, 1]
]

nowe_gradienty = [[1,-2],
                  [1,-1]]

batch = 4
suma=0
for i in range(len(gradienty)):
    for j in range(len(gradienty[i])):
        gradienty[i][j] += nowe_gradienty[i][j]
        
for i in range(len(gradienty)):
    for j in range(len(gradienty[i])): 
        gradienty[i][j] /=batch
