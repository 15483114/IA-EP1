import csv
import random
from random import shuffle

import numpy


def embaralho_tudo():
    file_teste = open('mnist_teste.csv')
    file_treinamento = open('mnist_treinamento.csv')
    treino = csv.reader(file_treinamento)
    teste = csv.reader(file_teste)

    combinacao = []

    for row in treino:
        combinacao.append(row)
    for row in teste:
        combinacao.append(row)

    shuffle(combinacao)

    partes = split_list(combinacao)

    for i in range(10):
        name = 'parte_' + str(i) + '.csv'
        m = open(name, 'w', newline='')
        with m:
            write = csv.writer(m)
            write.writerows(partes[i])


def split_list(alist, wanted_parts=10):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


embaralho_tudo()

size = 0
learn_rate = 0.1
epocas = 3

acuracia_treinamento = []
acuracia_teste = []
x = []

for i in range(epocas):
    x.append(int(i))


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


def train(file):
    global acuracia_treinamento
    exemplos = 600
    weights = [None] * 10
    obtido = [None] * 10
    esperado = [None] * 10
    erro = [None] * 10
    matriz_confusao = [[0 for i in range(10)] for j in range(10)]
    total = 0

    for c in range(10):
        weights[c] = create_weights()
    percentual_acerto = 0
    acertos = 0
    errado = 0
    for epoca in range(epocas):
        with open(file) as file:
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

                if esperado[line[0]] == obtido[line[0]]:
                    acertos += 1
                if esperado[line[0]] != obtido[line[0]]:
                    errado += 1

                total = errado + acertos
                percentual_acerto = acertos / total

                line = file.readline()

                # imprime_matriz(matriz_confusao)
        print('%d - acuracia treinamento: %f' % (epoca, percentual_acerto))
        acuracia_treinamento.append(percentual_acerto)
        test(weights, epoca)
    # print(acertos)
    # print(total)
    # imprime_matriz(matriz_confusao)
    # print(acertos)

    return weights


def train_kfold():
    global acuracia_treinamento
    exemplos = 100
    weights = [None] * 10
    obtido = [None] * 10
    esperado = [None] * 10
    erro = [None] * 10
    matriz_confusao = [[0 for i in range(10)] for j in range(10)]

    lista_acuracias = []

    for t in range(10):
        acuracia = 0
        acertos = 0
        errado = 0

        for c in range(10):
            weights[c] = create_weights()

        for epoca in range(epocas):
            for treino in range(10):
                if treino != t:
                    name = 'parte_' + str(treino) + '.csv'
                    with open(name) as file:
                        line = file.readline()

                        for teste in range(exemplos):
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

                            if esperado[line[0]] == obtido[line[0]]:
                                acertos += 1
                            if esperado[line[0]] != obtido[line[0]]:
                                errado += 1

                            total = errado + acertos
                            acuracia = acertos / total

                            line = file.readline()

                            # imprime_matriz(matriz_confusao)
                print('%d - acuracia treinamento: %f' % (epoca, acuracia))
                acuracia_treinamento.append(acuracia)
            name = 'parte_' + str(t) + '.csv'
            test(weights, epoca, name)


def test(weights, epoca, file):
    exemplos = 100
    percentual_acerto = 0
    acertos = 0
    obtido = [None] * 10
    esperado = [None] * 10
    global acuracia_teste

    matriz_confusao = [[0 for i in range(10)] for j in range(10)]

    # imprime_matriz(matriz_confusao)
    with open(file) as file:
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

    print('%d - acuracia teste: %f\n' % (epoca, percentual_acerto))
    acuracia_teste.append(percentual_acerto)


# holdout
# train('mnist_treinamento.csv')

# kfold
train_kfold()

import matplotlib.pyplot as pyplot

pyplot.plot(x, acuracia_treinamento, label='Treinamento')
pyplot.plot(x, acuracia_teste, label='Teste')

pyplot.xlabel('Epoca')
pyplot.ylabel('Acuracia')

pyplot.title('Holdout')

pyplot.legend()

pyplot.show()
# exit()
