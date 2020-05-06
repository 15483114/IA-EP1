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
    for i in range(len(array)):
        weights.append(random.uniform(-0.5, 0.5))
    print(weights)


def train(array, file_path):
    import random
    weights = []
    for i in range(len(array)):
        weights.append(random.uniform(-0.5, 0.5))

    n_epoch = 50
    l_rate = 0.1

    for epoch in range(n_epoch):
        sum_error = 0.0
        with open(file_path, 'r') as file:
            for k in range(1000):
                line = file.readline()

                if len(line) > 100:
                    line = line.split(',')
                    perceptron0 = weights[0]
                    perceptron1 = weights[0]
                    perceptron2 = weights[0]
                    perceptron3 = weights[0]
                    perceptron4 = weights[0]
                    perceptron5 = weights[0]
                    perceptron6 = weights[0]
                    perceptron7 = weights[0]
                    perceptron8 = weights[0]
                    perceptron9 = weights[0]

                    for i in range(1, len(line)):
                        perceptron0 += float(line[i]) / 255 * weights[i]
                        perceptron1 += float(line[i]) / 255 * weights[i]
                        perceptron2 += float(line[i]) / 255 * weights[i]
                        perceptron3 += float(line[i]) / 255 * weights[i]
                        perceptron4 += float(line[i]) / 255 * weights[i]
                        perceptron5 += float(line[i]) / 255 * weights[i]
                        perceptron6 += float(line[i]) / 255 * weights[i]
                        perceptron7 += float(line[i]) / 255 * weights[i]
                        perceptron8 += float(line[i]) / 255 * weights[i]
                        perceptron9 += float(line[i]) / 255 * weights[i]

                    if (activation > 0):
                        prediction = 1
                    else:
                        prediction = 0

                    if ((float(line[0]) == 7 and prediction == 1) or (float(line[0]) != 7) and prediction == 0):
                        error = 0
                    else:
                        error = 1

                    sum_error += error ** 2
                    weights[0] = weights[0] + l_rate * error

                    for i in range(1, len(line)):
                        weights[i] = weights[0] + l_rate * error * float(line[i]) / 255

        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        print(weights)
