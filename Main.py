import random

import numpy 


size = 0
learn_rate = 0.1
epocas = 15


def recebe_linha(line):
    line = line.split(',')
    line[784] = 0
    return line


def pre_processamento(line):
    for i in range(len(line) - 1):
        line[i] = float(line[i]) / 255
    return line


def define_resposta(line,ativacao):
    if (line == ativacao):
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
    return ativacao

def compara_ativacao(ativacoes):
    maior=max(ativacoes)
    return maior

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
    x = 30
    tentativas=0
    acertos=0
    print('TREINANDO')
    weights=[[],[],[],[],[],[],[],[],[],[]]
    erro=[[],[],[],[],[],[],[],[],[],[]]
    ativacao=[[],[],[],[],[],[],[],[],[],[]]
    for c in range (10):
        weights[c] = create_weights()
    for i in range(epocas):
        percentual_acerto = 0
        with open('mnist_treinamento.csv') as file:
            line = file.readline()
            for j in range(x):
                
                line = recebe_linha(line)
                line = pre_processamento(line)
                #answer = define_resposta(line)
                maior=0
                indice=0
                for l in range(10):
                    ativacao[l] = calcula_ativacao(weights[l], line)
                    if (ativacao[l]>maior):
                        maior=ativacao[l]
                        indice=l
                answer=define_resposta(line[0],maior)
                acertos+=answer
                tentativas+=1
#----------------------------------------------------------------                
                for m in range(10):
                    resposta=define_resposta(line[0],ativacao[l])
                    erro[m] = calcula_erro(resposta, 0)
                    weights[m] = atualiza_pesos(erro[m], weights[m], line)
                erro[indice]=calcula_erro(maior, answer)
                weights[indice] = atualiza_pesos(erro[indice], weights[indice], line)
                
                #percentual_acerto += erro[indice]
                percentual_acerto= acertos/tentativas
                # print('%d %f' % (answer, ativacao))
                line = file.readline()
                # learn_rate = 1 / (1 + numpy.exp(-erro))
        # print(weights)
        print('acuracia:%f' % (percentual_acerto))
        '''if abs(percentual_acerto) < 234:
            break '''
    return weights




def test(weights,i):
    print('TESTANDO Perceptron')
    print (i)
    percentual_acerto = 0
    with open('mnist_teste.csv') as file:
        line = file.readline()
        for j in range(1):
            line = recebe_linha(line)

            line = pre_processamento(line)

            answer = define_resposta(line)

            ativacao = calcula_ativacao(weights, line)

            erro = calcula_erro(ativacao, answer)

            print('%d %f' % (answer, ativacao))
            line = file.readline()

#weights = train(1)
#test(weights,1)
train()


#exit()
