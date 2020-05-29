import csv
import random
from random import shuffle

import matplotlib.pyplot as pyplot
import numpy

size = 0
learn_rate = 0.01
epocas = 20
exemplos = 100
acuracia_treinamento = []
acuracia_teste = []


def main():
    # embaralho_tudo()
    # treino_holdout('mnist_treinamento.csv', 'mnist_teste.csv',[0], True)
    treino_cross_validation()
    exit()


def treino_holdout(path, path_test,x, holdout):
    global acuracia_teste
    global acuracia_treinamento
    global exemplos
    acuracia_teste = []
    print('Iniciando Holdout')
    acuracia_treinamento = []
    erro, esperado, obtido, w = cria_arrays_auxiliares()
    matriz_confusao = cria_matriz_confusao()
    acuracia = 0
    acuracia_t = 0
    weights = [None] * 10

    if holdout == True:
        weights = w
        cria_pesos(weights)
    else:
        weights = x

    for epoca in range(epocas):
        acertos = 0
        errado = 0
        acuracia = test(weights, path)
        acuracia_t = test(weights, path_test)
        acuracia_treinamento.append(float(acuracia))
        acuracia_teste.append(acuracia_t)
        print('\n%d - acuracia treinamento: %f' % (epoca, acuracia))
        print('%d - acuracia teste: %f' % (epoca, acuracia_t))
        with open(path, 'r') as file:
            line = file.readline()

            for exemplo in range(exemplos):
                line = recebe_linha(line)
                line = pre_processamento(line)
                maior = 0
                indice = 0
                indice = calcula_ativacoes(indice, line, maior, obtido, weights)
                ativa_array_obtidos(indice, obtido)
                define_esperados(esperado, line)
                atualiza_matriz_confusao(line, matriz_confusao, obtido)
                weights = calcula_erro_atualiza_peso(erro, esperado, line, obtido, weights)
                acertos, errado = contabiliza_acertos(acertos, errado, esperado, line, obtido)
                # acuracia = calcula_acuracia(acertos, errado, acuracia)
                line = file.readline()

        if acuracia >= 0.98 or epoca == epocas - 1:
            acuracia = test(weights, path_test)
            acuracia_t = test(weights, path_test)
            acuracia_treinamento.append(float(acuracia))
            acuracia_teste.append(acuracia_t)
            break

    # imprime_resultado_holdout()
    # imprime_matriz(matriz_confusao)
    #
    # pyplot.matshow(matriz_confusao, cmap=pyplot.cm.plasma)
    # pyplot.xticks(numpy.arange(10))
    # pyplot.yticks(numpy.arange(10))
    # pyplot.title('Matriz de Confusão\n')
    # for i in range(10):
    #     for j in range(10):
    #         pyplot.text(j, i, matriz_confusao[i][j], color='white', va='center', ha='center')
    # pyplot.show()
    return acuracia, acuracia_t, weights


def treino_cross_validation():
    global epocas, exemplos, acuracia_treinamento, acuracia_teste
    acuracia_teste = []
    acuracia_treinamento = []
    for teste in range(10):
        acuracia = 0
        acuracia_t = 0
        weights = [None] * 10
        cria_pesos(weights)
        for treino in range(10):
            if treino != teste:
                print(str(teste) * 10)
                train_path = 'parte_' + str(treino) + '.csv'
                teste_path = 'parte_' + str(teste) + '.csv'
                a, b,weights = treino_holdout(train_path, teste_path, weights, False)
                acuracia += a
                acuracia_t += b
        acuracia_treinamento.append(acuracia)
        acuracia_teste.append(acuracia_t)

    media_teste = 0
    for i in acuracia_teste:
        media_teste += i
    media_teste /= 10

    media_treino = 0
    for i in acuracia_treinamento:
        media_treino += i
    media_treino /= 10
    print('+--------------------------------------------------------+')
    print('|Acuracia média de treinamento    Acuracia média de teste|')
    print('|%.3f                            %.3f         ' % (media_treino, media_teste))
    print('+--------------------------------------------------------+')


def update_accuracy_list(acuracia_teste_cross_validation, acuracia_treinamento_cross_validation, lista_acuracias_teste,
                         lista_acuracias_treinamento):
    lista_acuracias_treinamento.append(
        acuracia_treinamento_cross_validation[len(acuracia_treinamento_cross_validation) - 1])
    lista_acuracias_teste.append(acuracia_teste_cross_validation[len(acuracia_teste_cross_validation) - 1])


def print_graph(acuracia_teste_cross_validation, acuracia_treinamento_cross_validation, epocas, exemplos, teste_fold):
    pyplot.plot(acuracia_treinamento_cross_validation, label='Treinamento')
    pyplot.plot(acuracia_teste_cross_validation, label='Teste')
    pyplot.xlabel('Epoca')
    pyplot.ylabel('Acuracia')
    pyplot.title('Cross-validation\nFold: %d    Epocas: %d   Exemplos: %d    Learn rate: %.1f' % (
        teste_fold, epocas, exemplos, learn_rate))
    pyplot.legend()
    pyplot.ylim([0, 1])
    pyplot.xlim([0, 50])
    pyplot.show()


def print_result(lista_acuracias_teste, lista_acuracias_treinamento):
    media_treinamento = calcula_media(lista_acuracias_treinamento)
    media_teste = calcula_media(lista_acuracias_teste)
    print('................................\nAcuracia treinamento média %f' % media_treinamento)
    print('Acuracia exemplo média %f\n.........................................' % media_teste)


def test(weights, file):
    global exemplos
    e = int(exemplos)
    acuracia = 0
    acertos = 0
    obtido = [None] * 10
    esperado = [None] * 10
    global acuracia_teste
    errado = 0

    with open(file) as file:
        line = file.readline()
        for j in range(e):
            line = recebe_linha(line)
            line = pre_processamento(line)
            maior = 0
            indice = 0
            indice = calcula_ativacoes(indice, line, maior, obtido, weights)
            ativa_array_obtidos(indice, obtido)
            define_esperados(esperado, line)
            acertos, errado = contabiliza_acertos(acertos, errado, esperado, line, obtido)
            acuracia = calcula_acuracia(acertos, errado, acuracia)
            line = file.readline()

    return acuracia


def imprime_resultado_holdout():
    x = cria_array_auxiliar(epocas)
    pyplot.plot(acuracia_treinamento, label='Treinamento')
    pyplot.plot(acuracia_teste, label='Teste')
    pyplot.xlabel('Epoca')
    pyplot.ylabel('Acuracia')
    pyplot.title('Holdout\nEpocas: %d   Exemplos: %d    Learn rate: %.3f' % (epocas, exemplos, learn_rate))
    pyplot.legend()
    pyplot.ylim([0, 1])
    pyplot.xlim([0, 50])
    pyplot.show()


def embaralho_tudo():
    file_teste, file_treinamento = abre_arquivos()
    reader_teste, reader_treinamento = cria_csv_reader(file_teste, file_treinamento)
    combinacao = []
    combina_arquivos_em_array(combinacao, reader_teste, reader_treinamento)
    shuffle(combinacao)
    partes = split_list(combinacao)
    salva_partes(partes)


def salva_partes(partes):
    for i in range(10):
        name = 'parte_' + str(i) + '.csv'
        m = open(name, 'w', newline='')
        with m:
            write = csv.writer(m)
            write.writerows(partes[i])


def combina_arquivos_em_array(combinacao, teste, treino):
    for row in treino:
        combinacao.append(row)
    for row in teste:
        combinacao.append(row)


def cria_csv_reader(file_teste, file_treinamento):
    treino = csv.reader(file_treinamento)
    teste = csv.reader(file_teste)
    return teste, treino


def abre_arquivos():
    file_teste = open('mnist_teste.csv')
    file_treinamento = open('mnist_treinamento.csv')
    return file_teste, file_treinamento


def split_list(alist, wanted_parts=10):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


def cria_array_auxiliar(e):
    x = []
    for i in range(e):
        x.append(int(i))
    return x


def recebe_linha(line):
    line = line.split(',')
    line[784] = 0
    return line


def pre_processamento(line):
    line[0] = int(line[0].replace('[', ''))
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
        return weights


def imprime_matriz(matriz):
    for linha in matriz:
        for elemento in linha:
            print("%2d" % elemento, end=" ")
        print()


def cria_matriz_confusao():
    matriz_confusao = [[0 for i in range(10)] for j in range(10)]
    return matriz_confusao


def cria_arrays_auxiliares():
    weights = [None] * 10
    obtido = [None] * 10
    esperado = [None] * 10
    erro = [None] * 10
    return erro, esperado, obtido, weights


def calcula_acuracia(acertos, errado, acuracia):
    total = errado + acertos
    acuracia = acertos / total
    return acuracia


def contabiliza_acertos(acertos, errado, esperado, line, obtido):
    if esperado[line[0]] == obtido[line[0]]:
        acertos += 1
    if esperado[line[0]] != obtido[line[0]]:
        errado += 1
    return acertos, errado


def calcula_erro_atualiza_peso(erro, esperado, line, obtido, weights):
    for perceptron in range(10):
        erro[perceptron] = calcula_erro(esperado[perceptron], obtido[perceptron])
        weights[perceptron] = atualiza_pesos(erro[perceptron], weights[perceptron], line)
    return weights


def atualiza_matriz_confusao(line, matriz_confusao, obtido):
    for perceptron in range(10):
        matriz_confusao[line[0]][perceptron] += obtido[perceptron]


def calcula_ativacoes(indice, line, maior, obtido, weights):
    for perceptron in range(10):
        obtido[perceptron] = calcula_ativacao(weights[perceptron], line)
        indice = atualiza_maior_valor_obtido(indice, maior, obtido, perceptron)
    return indice


def define_esperados(esperado, line):
    for perceptron in range(10):
        esperado[perceptron] = define_resposta(int(line[0]), perceptron)


def ativa_array_obtidos(indice, obtido):
    for perceptron in range(10):
        if perceptron == indice:
            obtido[perceptron] = 1
        else:
            obtido[perceptron] = 0


def atualiza_maior_valor_obtido(indice, maior, obtido, perceptron):
    if (obtido[perceptron] > maior):
        maior = obtido[perceptron]
        indice = perceptron
    return indice


def cria_pesos(weights):
    for c in range(10):
        weights[c] = create_weights()


def calcula_media(array):
    sum = 0
    for i in array:
        sum += i
    return sum / 10


def define_nome_arquivo_abrir(treino):
    name = 'parte_' + str(treino) + '.csv'
    return name


main()
