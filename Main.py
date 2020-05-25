import random

import numpy

import csv
f = open('mnist_teste.csv')
f2 = open('mnist_treinamento.csv')
treino = csv.reader(f2)
teste = csv.reader(f)

# separando as linhas
xs=[]
combinacao=[]

for row in treino:
    xs.append(row)
    combinacao.append(row)

ts=[]
for row in teste:
    ts.append(row)
    combinacao.append(row)
#juntando os dois conjuntos em combinacao

# embaralhando um array do tamanho dos dois conjuntos combinados. O objetivo é utilizar esse array no indice do conjunto, para seguir a ordem embaralhada.
from random import shuffle

shuffle(combinacao)

m = open('misturinha.csv','w',newline='')

# writing the data into the file 
with m:     
    write = csv.writer(m) 
    write.writerows(combinacao)



size = 0
learn_rate = 0.1
epocas = 10


def recebe_linha(line):
    line = line.split(',')
    line[784] = 0
    return line


def pre_processamento(line):
    line[0] = int(line[0])
    for i in range(1, len(line) - 1):
        line[i] = float(line[i]) / 255
    return line


def define_resposta(resposta_correta, obtido):
    if resposta_correta == obtido:
        answer = 1
    else:
        answer = 0
    return answer


def inicializa_pesos_aleatorios(weights):
    for i in range(size):
        weights[i] = random.uniform(-0.5, 0.5)
    return weights


def calcula_ativacao(weights, line):
    # print(resposta_correta)
    ativacao = weights[0]
    for i in range(1, size):
        ativacao += weights[i] * line[i]
    return ativacao


def compara_ativacao(ativacoes):
    maior = max(ativacoes)
    return maior


def calcula_erro(answer, ativacao):
    erro = answer - ativacao
    return erro


def atualiza_pesos(erro, weights, line):
    weights[0] += learn_rate * erro
    for i in range(1, size):
        weights[i] += learn_rate * erro * line[i]
    return weights


def create_weights():
    global size
    with open('mnist_treinamento.csv') as file:
        line = file.readline()
        line = recebe_linha(line)
        size = len(line)
        weights = numpy.zeros(size)
        weights = inicializa_pesos_aleatorios(weights)
        # print(weights)
        return weights


def imprime_matriz(matriz):
    for linha in matriz:
        for elemento in linha:
            print("%2d" % elemento, end=" ")
        print()


def train():
    exemplos = 1000
    print('TREINANDO')
    weights = [None] * 10
    obtido = [None] * 10
    esperado = [None] * 10
    erro = [None] * 10
    matriz_confusao = [[0 for i in range(10)] for j in range(10)]
    total=0
    
    #imprime_matriz(matriz_confusao)

    for c in range(10):
        weights[c] = create_weights()
    percentual_acerto = 0
    acertos = 0
    errado=0 
    for epoca in range(epocas):

        with open('mnist_treinamento.csv') as file:
            line = file.readline()

            for j in range(exemplos):
                line = recebe_linha(line)
                line = pre_processamento(line)

                maior = 0
                indice = 0

                for l in range(10):
                    obtido[l] = calcula_ativacao(weights[l], line)
                    if (obtido[l] > maior):
                        maior = obtido[l]
                        indice = l

                for perceptron in range(10):
                    if perceptron == indice:
                        obtido[perceptron] = 1
                    else:
                        obtido[perceptron] = 0

                for perceptron in range(10):
                    esperado[perceptron] = define_resposta(int(line[0]), perceptron)

                for perceptron in range(10):
                    matriz_confusao[line[0]][perceptron] += obtido[perceptron]
                    

                for perceptron in range(10):
                    erro[perceptron] = calcula_erro(esperado[perceptron], obtido[perceptron])
                    weights[perceptron] = atualiza_pesos(erro[perceptron], weights[perceptron], line)


                # print()
                # print(esperado)
                # print(obtido)
                # print(erro)
                # print()

                if esperado[line[0]] == obtido[line[0]]:
                    acertos += 1
                if esperado[line[0]] != obtido[line[0]]:
                    errado += 1
                    
                total=errado+acertos
                percentual_acerto = acertos / total

                line = file.readline()
                print('acuracia:%f' % (percentual_acerto))
                imprime_matriz(matriz_confusao)
     
    print("FINAL")
    print(acertos)
    print(total)
    print('acuracia:%f' % (percentual_acerto))
    imprime_matriz(matriz_confusao)

    

    return weights


def test(weights):
    print('TESTANDO Perceptron')
    exemplos = 1000
    percentual_acerto = 0
    acertos = 0
    obtido = [None] * 10
    esperado = [None] * 10

    matriz_confusao = [[0 for i in range(10)] for j in range(10)]


    with open('mnist_teste.csv') as file:
        line = file.readline()
        for j in range(exemplos):
                line = recebe_linha(line)
                line = pre_processamento(line)

                maior = 0
                indice = 0

                for l in range(10):
                    obtido[l] = calcula_ativacao(weights[l], line)
                    if (obtido[l] > maior):
                        maior = obtido[l]
                        indice = l

                for perceptron in range(10):
                    if perceptron == indice:
                        obtido[perceptron] = 1
                    else:
                        obtido[perceptron] = 0

                for perceptron in range(10):
                    esperado[perceptron] = define_resposta(int(line[0]), perceptron)

                for perceptron in range(10):
                    matriz_confusao[line[0]][perceptron] += obtido[perceptron]


                if esperado[line[0]] == obtido[line[0]]:
                    acertos += 1


                percentual_acerto = acertos / exemplos

                line = file.readline()

    print('acuracia:%f' % (percentual_acerto))
    imprime_matriz(matriz_confusao)


weights = train()
test(weights)
#train()

# exit()
