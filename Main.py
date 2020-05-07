import numpy


def size(file_path):
    with open(file_path, 'r') as file:
        size = 0
        while (1):
            if (file.readline() != ''):
                size += 1
            else:
                break;
        print('number of lines: %d' % size)


def createWeights(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        array = line.split(',')
    print(len(array))
    import random
    weights = []
    for i in range(len(array) - 1):
        weights.append(random.uniform(-0.5, 0.5))
    print(weights)


def train(file_path):
    import random
    with open(file_path, 'r') as file:
        line = file.readline()
        array = line.split(',')
    weights = numpy.zeros((10, len(array)))
    for i in range(9):
        for j in range(len(array) - 1):
            weights[i][j] = (random.uniform(-0.5, 0.5))

    n_epoch = 10
    learn_rate = 0.1

    for epoch in range(n_epoch):
        sum_error = numpy.zeros(10)
        error_p = numpy.zeros(10)
        with open(file_path, 'r') as file:
            # quantas linhas vamos analisar
            for k in range(10):
                line = file.readline()

                if len(line) > 10:
                    line = line.split(',')
                    perceptron = numpy.zeros(10)

                    for i in range(10):
                        perceptron[i] = weights[i][0]

                    for g in range(10):
                        for i in range(1, len(line)):
                            perceptron[g] += float(line[i]) / 255 * weights[g][i]

                    error = numpy.zeros(10)

                    for i in range(10):
                        if int(line[0]) == i:
                            answer = 1
                        else:
                            answer = 0

                        perceptron[i] = 1 / (1 + numpy.exp(-perceptron[i]))

                        error[i] = -(answer - perceptron[i])
                        # sum_error[i] += error[i]
                        weights[i][0] = weights[i][0] + learn_rate * error[i]

                        if error[i] >= 0.5:
                            error_p[i] += 1

                        for j in range(1, len(line)):
                            weights[i][j] = weights[i][0] + learn_rate * error[i] * float(line[j]) / 255

        print(error_p)

    return weights


def test(file_path, weights):
    with open(file_path, 'r') as file:
        for k in range(10):
            line = file.readline()
            if len(line) > 10:
                line = line.split(',')
                perceptron = numpy.zeros(10)

                for i in range(10):
                    perceptron[i] = weights[i][0]

                for g in range(10):
                    for i in range(1, len(line)):
                        perceptron[g] += float(line[i]) / 255 * weights[g][i]

                result = 0
                for i in range(len(perceptron)):
                    perceptron[i] = 1 / (1 + numpy.exp(-perceptron[i]))
                    if perceptron[result] < perceptron[i]:
                        result = i

                print(perceptron)
                print('correta: %d, resultado: %d' % (float(line[0]), result))


# file = open("weights.txt", "x")
weights = train('mnist_treinamento.csv')
# numpy.savetxt(file, )
# file.close()

# file.open('weights.txt', 'r')
# weights = file.readline()
# test('mnist_teste.csv', weights)
exit()
