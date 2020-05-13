import random

import numpy

size = 0
learn_rate = 0.1
epocas = 50


def recebe_linha(line):
    line = line.split(',')
    line[784] = 0
    return line


def pre_processamento(line):
    for i in range(len(line) - 1):
        line[i] = float(line[i]) / 255
    return line


def define_resposta(line):
    if line[0] == 0:
        answer = 1
    else:
        answer = 0
    return answer


def inicializa_pesos_aleatorios(weights):
    for i in range(size):
        weights[i] = random.uniform(-0.5, 0.5)
    return weights


def calcula_ativacao(weights, line):
    # print(line)
    ativacao = weights[0]
    for i in range(size):
        ativacao += weights[i] * line[i]
    ativacao = 1 / (1 + numpy.exp(-ativacao))
    return ativacao


def calcula_erro(ativacao, answer):
    # erro = numpy.power((answer - ativacao), 2)
    erro = answer - ativacao
    return erro


def atualiza_pesos(erro, weights, line):
    weights[0] += learn_rate * erro
    for i in range(size):
        weights[i] += learn_rate * erro * line[i] + 0.0001
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


def train():
    x = 60000
    print('TREINANDO')
    weights = create_weights()
    for i in range(epocas):
        percentual_acerto = 0
        with open('mnist_treinamento.csv') as file:
            line = file.readline()
            for j in range(x):
                line = recebe_linha(line)
                line = pre_processamento(line)
                answer = define_resposta(line)
                ativacao = calcula_ativacao(weights, line)
                erro = calcula_erro(ativacao, answer)
                weights = atualiza_pesos(erro, weights, line)
                percentual_acerto += erro
                # print('%d %f' % (answer, ativacao))
                line = file.readline()
                # learn_rate = 1 / (1 + numpy.exp(-erro))
        # print(weights)
        print('erro:%f' % (percentual_acerto))
        if abs(percentual_acerto) < 234:
            break
    return weights


def test(weights):
    print('TESTANDO')
    percentual_acerto = 0
    with open('mnist_teste.csv') as file:
        line = file.readline()
        for j in range(10000):
            line = recebe_linha(line)

            line = pre_processamento(line)

            answer = define_resposta(line)

            ativacao = calcula_ativacao(weights, line)

            erro = calcula_erro(ativacao, answer)

            print('%d %f' % (answer, ativacao))
            line = file.readline()

weights = train()
test(weights)
exit()
