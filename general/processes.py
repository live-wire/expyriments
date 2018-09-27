# Processes maxing out

from multiprocessing import Process
import numpy as np
import os
import time

def func():
	for i in range(4000000):
		np.sqrt(i)

time1 = time.time()
processes = []

for core in range(os.cpu_count()):
	print("Registering Process %d"%core)
	processes.append(Process(target=func))

for process in processes:
	process.start()

for process in processes:
	process.join()

print("Elapsed time = ", time.time() - time1)
