from scipy import signal

import os
os.system('cls')

entrada = [
   [1.0, 2.0, 3.0, 4.0],
   [5.0, 6.0, 7.0, 8.0],
   [4.0, 3.0, 2.0, 1.0],
   [8.0, 7.0, 6.0, 5.0],
]

filtro = [
   [3.0, 2.0, 1.0],
   [1.0, 2.0, 3.0],
   [4.0, 5.0, 6.0]
]

gradiente = [
   [9, 8, 7],
   [3, 2, 1],
   [3, 4, 6],
]

saida = signal.correlate2d(entrada, filtro, "valid")
gradK = signal.correlate2d(entrada, gradiente, "valid")
gradE = signal.convolve2d(gradiente, filtro, "full")

print('saida\n', saida)
print('\ngrad kernel\n', gradK)
print('\ngrad entrada\n', gradE)