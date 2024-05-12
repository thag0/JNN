import numpy as np
import timeit
import os

os.system('cls')

linA = 10
colB = 10
k = 100

a = np.random.randn(linA, k)
b = np.random.randn(k, colB)

def mult():
    return a @ b

tempo = timeit.timeit(mult, number=10000)
tempo_nano = tempo / 10000 * 1e9

print("Tempo médio de multiplicação em nanossegundos:", tempo_nano)