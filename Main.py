import csv
import random
from random import shuffle

import matplotlib.pyplot as pyplot
import numpy

size = 0
learn_rate = 0.1
epocas = 3
acuracia_treinamento = []
acuracia_teste = []

acuracia_treinamento_cross_validation = []
acuracia_teste_cross_validation = []

x = []


def main():
    # embaralho_tudo()
    # treino_holdout('mnist_treinamento.csv')
    treino_cross_validation()
    exit()


def imprime_resultado_holdout():
    cria_array_auxiliar(epocas)
    pyplot.plot(x, acuracia_treinamento, label='Treinamento')
    pyplot.plot(x, acuracia_teste, label='Teste')
    pyplot.xlabel('Epoca')
    pyplot.ylabel('Acuracia')
    pyplot.title('Holdout')
    pyplot.legend()
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


def treino_holdout(path):
    global acuracia_teste
    acuracia_teste = []
    print('Iniciando Holdout')
    global acuracia_treinamento
    exemplos = 600
    erro, esperado, obtido, weights = cria_arrays_auxiliares()
    matriz_confusao = cria_matriz_confusao()

    cria_pesos(weights)

    for epoca in range(epocas):
        acuracia = 0
        acertos = 0
        errado = 0
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
                calcula_erro_atualiza_peso(erro, esperado, line, obtido, weights)
                acertos, errado = contabiliza_acertos(acertos, errado, esperado, line, obtido)
                acuracia = calcula_acuracia(acertos, errado, acuracia)
                line = file.readline()

        print('%d - acuracia treinamento: %f' % (epoca, acuracia))
        acuracia_treinamento.append(acuracia)
        test(weights, epoca, 'mnist_teste.csv')

    imprime_resultado_holdout()


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


def treino_cross_validation():
    global acuracia_teste
    print('...................\nIniciando Kfold Cross-Validation\n........................')
    global acuracia_treinamento_cross_validation
    exemplos = 3000
    weights = [None] * 10
    obtido = [None] * 10
    esperado = [None] * 10
    erro = [None] * 10
    matriz_confusao = [[0 for i in range(10)] for j in range(10)]

    lista_acuracias_teste = []
    lista_acuracias_treinamento = []

    for teste_fold in range(10):
        print('...................\nFold de teste: %d\n........................' % teste_fold)
        acuracia_teste = []
        x = []
        acuracia = 0
        acertos = 0
        errado = 0
        epocas_kfold = 6

        for perceptron in range(10):
            weights[perceptron] = create_weights()
        acuracia_treinamento_cross_validation = []
        for epoca in range(epocas_kfold):
            for treino in range(10):
                if treino != teste_fold:
                    path_file = define_nome_arquivo_abrir(treino)

                    with open(path_file) as file:
                        line = file.readline()

                        for teste in range(exemplos):
                            line = recebe_linha(line)
                            line = pre_processamento(line)

                            maior = 0
                            indice = 0

                            indice = calcula_ativacoes(indice, line, maior, obtido, weights)
                            ativa_array_obtidos(indice, obtido)
                            define_esperados(esperado, line)
                            atualiza_matriz_confusao(line, matriz_confusao, obtido)
                            calcula_erro_atualiza_peso(erro, esperado, line, obtido, weights)
                            acertos, errado = contabiliza_acertos(acertos, errado, esperado, line, obtido)
                            acuracia = calcula_acuracia(acertos, errado, acuracia)
                            line = file.readline()

            print('Fold %d - Epoca %d - Acuracia treinamento %f' % (teste_fold, epoca, acuracia))
            acuracia_treinamento_cross_validation.append(acuracia)
            path_file = define_nome_arquivo_abrir(teste_fold)
            test(weights, teste_fold, path_file)

        x = []
        x = cria_array_auxiliar(epocas_kfold)
        print(x)
        pyplot.plot(x, acuracia_treinamento_cross_validation, label='Treinamento')
        pyplot.plot(x, acuracia_teste, label='Teste')
        pyplot.xlabel('Epoca')
        pyplot.ylabel('Acuracia')
        pyplot.title('Cross-Validation\nFold %d' % teste_fold)
        pyplot.legend()
        pyplot.show()
        pyplot.close()
        lista_acuracias_treinamento.append(
            acuracia_treinamento_cross_validation[len(acuracia_treinamento_cross_validation) - 1])
        lista_acuracias_teste.append(acuracia_teste[len(acuracia_teste) - 1])

    media_treinamento = calcula_media(lista_acuracias_treinamento)
    media_teste = calcula_media(lista_acuracias_teste)

    print('................................\ncuracia treinamento média %f' % media_treinamento)
    print('Acuracia teste média %f\n.........................................' % media_teste)


def define_nome_arquivo_abrir(treino):
    name = 'parte_' + str(treino) + '.csv'
    return name


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


main()
