import numpy


def train(file_path):
    weights = create_weights(file_path)
    n_epoch = 100
    learn_rate = 0.01

    for epoch in range(n_epoch):
        acertos = 0
        with open(file_path, 'r') as file:
            for k in range(100):
                row = file.readline()

                if len(row) > 1:
                    row = row.split(',')
                    activation = numpy.zeros(10)
                    error = numpy.zeros(10)
                    row = pre_process(row)

                    for i in range(10):
                        calculate_activation(i, row, activation, weights)
                        answer = calculate_error(error, i, row, activation)
                        update_weights(error, i, learn_rate, row, weights)
                        # if i == 0:
                        #     print(
                        #         'activation:%f answer:%f error:%f' % (activation[i], answer, error[i]))

                    result = 0
                    for i in range(10):
                        if (activation[result] > activation[i]):
                            result = i
                    # print(result)
                    if result == float(row[0]):
                        acertos += 1
        print('epoch:%d acertos:%d' % (epoch, acertos))

        # print(weights[0][300])

    return weights


def update_weights(error, i, learn_rate, row, weights):
    weights[i][0] = weights[i][0] + learn_rate * error[i]
    for j in range(1, len(row)):
        weights[i][j] = weights[i][j] + learn_rate * error[i] * float(row[i])


def calculate_activation(i, line, activation, weights):
    activation[i] = weights[i][0]
    for g in range(1, len(line)):
        activation[i] += line[g] * weights[i][g]


def calculate_error(error, i, line, perceptron):
    if int(line[0]) == i:
        answer = 1
    else:
        answer = 0
    # perceptron[i] = 1 / (1 + numpy.exp(-perceptron[i]))
    # error[i] = answer - perceptron[i]
    # if answer == 0:
        # error[i] /= 2
    if answer == 1:
        error[i] = answer - perceptron[i]
        error[i] *= 100
    return answer


def create_weights(file_path):
    import random
    with open(file_path, 'r') as file:
        line = file.readline()
        array = line.split(',')
    weights = numpy.zeros((10, len(array)))
    for i in range(9):
        for j in range(len(array)):
            weights[i][j] = (random.uniform(-0.5, 0.5))
    return weights


def pre_process(line):
    for i in range(1, len(line)):
        line[i] = float(line[i]) / 255
    return line


def test(file_path, weights):
    with open(file_path, 'r') as file:
        for k in range(100):
            row = file.readline()

            if len(row) > 1:
                row = row.split(',')
                perceptron = numpy.zeros(10)
                error = numpy.zeros(10)
                row = pre_process(row)

                for i in range(10):
                    calculate_activation(i, row, perceptron, weights)

                result = 0
                for i in range(10):
                    if (perceptron[result] > perceptron[i]):
                        result = i

                # print('%d -- %d' % (int(row[0]), result))


weights = train('mnist_treinamento.csv')

test('mnist_teste.csv', weights)

exit()
