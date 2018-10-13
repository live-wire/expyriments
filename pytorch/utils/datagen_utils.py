# Data generator

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

# Spiral Data generator
# Credits: Andrej Karpathy http://cs231n.github.io/neural-networks-case-study/
class Spiral:
	def __init__(self, D=2, K=3):
		self.D = D
		self.K = K
	def generate(self, display=False):
		X = np.zeros((self.N * self.K, self.D)) # data matrix (each row = single example)
		y = np.zeros(self.N * self.K, dtype='uint8') # class labels
		for j in range(self.K):
		  ix = range(self.N * j, self.N * (j + 1))
		  r = np.linspace(0.0,1, self.N) # radius
		  t = np.linspace(j*4,(j+1)*4, self.N) + np.random.randn(self.N)*0.2 # theta
		  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		  y[ix] = j
		if display:
			plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
			plt.show()
		return X,y

class Gaussian:
	def __init__(self, mean, cov, n):
		self.mean = mean
		self.cov = cov
		self.n = n
	def generate(self):
		return np.random.normal(self.mean, self.cov, self.n)

class Sine:
	def __init__(self, Fs=8000, f = 5, sample = 8000):
		self.Fs = Fs
		self.f = f
		self.sample = sample

	def generate(self, display=False):
		Fs = self.Fs
		f = self.f
		sample = self.sample
		x = np.arange(sample)
		y = np.sin(2 * np.pi * f * x / Fs)
		return x,y