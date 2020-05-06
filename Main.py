def trainSize():
    with open('/content/mnist_treinamento.csv', 'r') as train:
        trainSize = 0
        while (1):
            if (train.readline() != ''):
                trainSize += 1
            else:
                break;
        print('size train dataset: %d' % (trainSize))

def testSize():
    with open('/content/mnist_teste.csv', 'r') as test:
        testSize = 0
        while (1):
            if (test.readline() != ''):
                testSize += 1
            else:
                break;
        print('size test dataset: %d' % (testSize))

def createWeights():
    with open('/content/mnist_treinamento.csv', 'r') as train:
        line = train.readline()
        array = line.split(',')
    print(len(array))
    import random
    weights = []
    for i in range(len(array)):
        weights.append(random.uniform(-0.5, 0.5))
    print(weights)

def trainPerceptron():
    import random
    weights = []
    for i in range(len(array)):
        weights.append(random.uniform(-0.5, 0.5))

    import numpy

    n_epoch = 50
    l_rate = 0.1

    for epoch in range(n_epoch):
        sum_error = 0.0
        with open('/content/mnist_treinamento.csv', 'r') as train:
            for k in range(1000):
                line = train.readline()

                if (len(line) > 100):
                    line = line.split(',')
                    activation = weights[0]

                    for i in range(1, len(line)):
                        activation0 += float(line[i]) / 255 * weights[i]
                        activation1 += float(line[i]) / 255 * weights[i]
                        activation2 += float(line[i]) / 255 * weights[i]
                        activation3 += float(line[i]) / 255 * weights[i]
                        activation4 += float(line[i]) / 255 * weights[i]
                        activation5 += float(line[i]) / 255 * weights[i]
                        activation6 += float(line[i]) / 255 * weights[i]
                        activation7 += float(line[i]) / 255 * weights[i]
                        activation8 += float(line[i]) / 255 * weights[i]
                        activation9 += float(line[i]) / 255 * weights[i]

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

