import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import interactive

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--net', help='Name of the folder that contains your net implementation (and contains intermediate/visualization_data.pt)')
parser.add_argument('--time', help='Plot update every n seconds')
args = parser.parse_args()

import sys
sys.path.append('..')

import threading 
import time

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
		# print(self.data.train_accuracy, self.data.test_accuracy, self.data.accuracy_epochs)
		

	def lossVsEpochs(self, data, model, epoch, criterion):
		self.data['loss_epochs'].append(epoch)
		from utils.deep_utils import getLoss
		self.data['train_loss'].append(getLoss(model, criterion, data[0], data[1]))
		torch.save(self.data, self.filename)
		# print(self.data['train_loss'], self.data['loss_epochs'])
		

	def loadData(self):
		try:
			print('Trying to load', self.filename)
			self.data = torch.load(self.filename)
		except:
			print('errored')
			self.data = {
				'train_accuracy': [],
				'test_accuracy': [],
				'accuracy_epochs': [],

				'train_loss': [],
				'loss_epochs': []
			}

	def showPlots(self):
		fig = plt.figure(1)
		fig.clear()
		plt.plot(self.data['accuracy_epochs'], self.data['train_accuracy'], label='training accuracy', color='blue')
		plt.plot(self.data['accuracy_epochs'], self.data['test_accuracy'], label='test accuracy', color='red')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy %')
		plt.pause(0.0001)
		plt.draw()

		fig = plt.figure(2)
		fig.clear()
		plt.plot(self.data['loss_epochs'], self.data['train_loss'], label='training loss', color='green')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Training Loss')
		plt.pause(0.0001)
		plt.draw()


def Visualize(filename = 'intermediate/visualization_data.pt'):
	_visualize = _Visualize(filename)
	return _visualize



def run_check():
	filepath = 'intermediate/visualization_data.pt'
	if args.net:
		filepath = args.net + '/' + filepath
	visualize = Visualize(filepath)
	visualize.showPlots()
	visualize = None
	plt.show(block=False)

if __name__ == '__main__':
	timer = 5
	if args.time:
		timer = float(args.time)
	while True:
		time.sleep(timer)
		run_check()


