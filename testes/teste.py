from keras.layers import MaxPool2D
import numpy as np
import os

os.system('cls')

entrada = np.array([
   [1, 2, 3],
   [4, 5, 6],
   [7, 8, 9],
])

entrada = entrada.reshape((1, 3, 3, 1))

camada = MaxPool2D((2, 2), 2)
saida = camada.call(entrada)