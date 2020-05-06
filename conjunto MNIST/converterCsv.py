import numpy

from numpy import genfromtxt
arrayCsv = genfromtxt('pré-processamento/mnist_teste.csv', delimiter=',')

from PIL import Image
im = Image.fromarray(arrayCsv)
im.save('pré-processamento/imagens/your_file.jpg')