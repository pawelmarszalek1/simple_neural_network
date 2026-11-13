from neural_network import *


df = pd.read_csv('test1.csv')
dane_wejsciowe= df.drop("cena", axis=1)#.values.tolist()
ceny = df['cena']


X_train, X_test, y_train, y_test = train_test_split(dane_wejsciowe,ceny,test=0.2,ziarno=42)

X_min = X_train.min()
X_max = X_train.max()

X_train_norm = minmax(X_train, X_min, X_max)
X_test_norm = minmax(X_test, X_min, X_max)

X_train_list = X_train_norm.values.tolist()
X_test_norm = X_test_norm.values.tolist()

wagi = stworz_wagi(X_train_list,1)
bias = stworz_biasy(1)

#z1, a1 = neuron(X_train_list[0],wagi,bias,relu)

print(X_train_list)
print(bias)
print(wagi)

#print(z1)

x = [[3,2,4][1,3,1],[7,1,2],[1,2,3]]
w = [3,2,5]
y = [35,17,35,26]

print(len(x))