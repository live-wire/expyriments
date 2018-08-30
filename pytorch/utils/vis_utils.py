import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

class _Visualize:
	def __init__(self, filename):
		self.filename = filename
		self.loadData()
		

	def accuracyVsEpochs(self, data, model, epoch):
		self.data['accuracy_epochs'].append(epoch)
		from utils.deep_utils import getAccuracy
		self.data['train_accuracy'].append(getAccuracy(model, data[0], data[1]))
		self.data['test_accuracy'].append(getAccuracy(model, data[2], data[3]))
		torch.save(self.data, self.filename)
		# print(self.train_accuracy, self.test_accuracy, self.accuracy_epochs)
		
		plt.figure(1)
		plt.plot(self.data['accuracy_epochs'], self.data['train_accuracy'], label='training accuracy', color='blue')
		plt.plot(self.data['accuracy_epochs'], self.data['test_accuracy'], label='test accuracy', color='red')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy %')
		plt.draw()
		plt.pause(0.0001)
		plt.clf()

	def loadData(self):
		try:
			print('Trying to load', self.filename)
			self.data = torch.load(self.filename)
		except:
			print('errored')
			self.data = {
				'train_accuracy': [],
				'test_accuracy': [],
				'accuracy_epochs': []
			}

def Visualize(filename = 'intermediate/visualization_data.pt'):
	_visualize = _Visualize(filename)
	return _visualize
