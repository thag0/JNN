import numpy as np
import time
import os

os.system('cls')

linA = 40
colB = 40
k = 40

a = np.random.randn(linA, k)
b = np.random.randn(k, colB)

start_time = time.perf_counter_ns()
res = a @ b
end_time = time.perf_counter_ns()

execution_time_ns = end_time - start_time
print("Tempo:", format(execution_time_ns, ',d'), "ns")