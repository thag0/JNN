import numpy as np
from scipy import signal
import time
import os

os.system('cls')

a = np.random.randn(28, 28)
b = np.random.randn(3, 3)

start_time = time.perf_counter_ns()
res = signal.correlate2d(a, b, "valid")
end_time = time.perf_counter_ns()

execution_time_ns = end_time - start_time
print("Tempo:", format(execution_time_ns, ',d'), "ns")