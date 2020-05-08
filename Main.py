import numpy


def train(file_path):
    weights = create_weights(file_path)
    n_epoch = 100
    learn_rate = 0.1

    for epoch in range(n_epoch):
        error0 = 0
        error1 = 0
        error2 = 0
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
                        #     print(activation[i])

                    error0 += error[0]
                    error1 += error[1]
                    error2 += error[2]
        print(
            '>epoch=%d, lrate=%.3f, error0=%.3f error1=%.3f error2=%.3f' % (epoch, learn_rate, error0, error1, error2))
    return weights


def update_weights(error, i, learn_rate, row, weights):
    weights[i][0] = weights[i][0] + learn_rate * error[i]
    for j in range(1, len(row)):
        weights[i][j] = weights[i][j] + learn_rate * error[i] * float(row[i])


def calculate_activation(i, line, perceptron, weights):
    perceptron[i] = weights[i][0]
    for g in range(1, len(line)):
        perceptron[i] += line[g] * weights[i][g]


def calculate_error(error, i, line, perceptron):
    if int(line[0]) == i:
        answer = 1
    else:
        answer = 0
    perceptron[i] = 1 / (1 + numpy.exp(-perceptron[i]))
    error[i] = answer - perceptron[i]
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
