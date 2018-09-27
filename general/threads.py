# Threads maxing out
from threading import Thread
import numpy as np
import os
import time

def func():
	for i in range(4000000):
		np.sqrt(i)
time1 = time.time()
threads = []

for core in range(os.cpu_count()):
	print("Thread registered:%d"%core)
	threads.append(Thread(target=func))

for thread in threads:
	thread.start()

for thread in threads:
	thread.join()

print("Elapsed time=", (time.time()-time1))