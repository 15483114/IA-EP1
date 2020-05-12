

import csv
f = open('mnist_teste.csv')
treino = csv.reader(f)

# separando as linhas
xs=[]
for row in treino:
    xs.append(row)

  
import random

def generate_weights():
    weights = []

    for i in range(0, 785):
        x = random.uniform(-0.5, 0.5)
        weights.append(x)
        
    return weights

def oe(x,w): # calcula o resultado obtido
    Soma=0 
    for i in range (1,783,1): #lenght? len(treino.iloc[1])    
        Soma=Soma+ x[i]*w[i]
        if Soma==x[i][0]:
            return 1
        else:
            return 0

def te(perceptronNumeroX, NumeroDoTreino): # fornece o resultado esperado
    if (perceptronNumeroX == NumeroDoTreino):
        return 1
    else:
        return 0

n=0.1
epoca=1

# iniciando a regra do perceptron
weights= []
for o in range(10):
    w= generate_weights()
    weights.append(w)


for k in range (epoca):
    for i in range (9999): #trocar por size of rows
        bias=[[], [] ,[] ,[] ,[] ,[] ,[], [], [], [] ] #bias/delta que ta no slide
        for j in range (1,785):
            for h in range(10):
                
                oE=oe(xs[j],weights[h])
                print(oE)
                tE= te(h,xs[i][0])
                bias[h][j]= n*(tE-oE)*xs[i][j]
                weights[h][j] = weights[h][j]+ bias
                print(bias[h])
                
    
