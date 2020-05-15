import numpy


def train(file_path):
    weights = create_weights(file_path)
    n_epoch = 50
    learn_rate = 1
    print('TRAIN')
    for epoch in range(n_epoch):
        sum_error = 0
        with open(file_path, 'r') as file:
            for k in range(1000):
                row = file.readline()

                if len(row) > 1:
                    row = row.split(',')
                    row[784] = 0
                    activation = numpy.zeros(10)
                    error = numpy.zeros(10)
                    row = pre_process(row)

                    for i in range(10):
                        activation = calculate_activation(i, row, activation, weights)

                    result = define_result(activation)

                    activation = activate_perceptrons(activation, result)

                    for i in range(10):
                        error = calculate_error(error, i, row, activation)
                        weights = update_weights(error, i, learn_rate, row, weights)

                    for i in error:
                        if i == 1:
                            sum_error += 1
                    # print('%d -> %d'%(float(row[0]),result))

        print('epoch:%d sum_error:%d' % (epoch, sum_error))
        # print(weights[1][294])
    return weights


def activate_perceptrons(activation, result):
    activation[result] = 1
    for i in range(10):
        if i != result:
            activation[i] = 0
    return activation


def define_result(activation):
    result = 0
    for i in range(10):
        if activation[result] < activation[i]:
            result = i
    return result


def update_weights(error, i, learn_rate, row, weights):
    weights[i][0] += learn_rate * error[i]
    for j in range(1, len(row)):
        weights[i][j] += learn_rate * error[i] * float(row[i])
    return weights


def calculate_activation(i, line, activation, weights):
    activation[i] = weights[i][0]
    for g in range(1, len(line)):
        activation[i] += line[g] * weights[i][g]
    return activation


def calculate_error(error, i, line, activation):
    if int(line[0]) == i:
        answer = 1
    else:
        answer = 0

    error[i] = answer - activation[i]
    return error


def create_weights(file_path):
    import random
    with open(file_path, 'r') as file:
        line = file.readline()
        array = line.split(',')
    weights = numpy.zeros((10, len(array)))
    for i in range(10):
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
