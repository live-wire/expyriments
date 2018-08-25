# Ten hand gestures recognition here
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Adding parent folder to path (Python3 workaround)
import sys
sys.path.append('..')

from dynamic_net import DynamicNet
import deep_utils as dutils
import preprocess_utils as prep



def main():
	x = np.load('data/X.npy')
	print(x.shape)
	x = prep.flattenRows(x)
	print(x.shape)
	y = np.load('data/Y.npy')
	print(y.shape)
	x_train, x_test, y_train, y_test = prep.splitDataset(x,y)
	print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

	N, D_in, D_H, N_H, D_out = 1381, 4096, 1000, 3, 10

	saved_state_filename = 'intermediate/%d_%d_%d_%d_DynamicNet.pt' % (D_in, D_H, N_H, D_out)


	# Tensors for the dataset
	x = torch.tensor(x_train).float()
	y = torch.tensor(y_train).float()
	x_test = torch.tensor(x_test).float()
	y_test = torch.tensor(y_test).float()


	# Construct our model by instantiating the class defined above
	model = DynamicNet(D_in, D_H, N_H, D_out)
	# Construct our loss function and an Optimizer. Training this strange model with
	# vanilla stochastic gradient descent is tough, so we use momentum

	criterion = torch.nn.MSELoss(reduction='sum')
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


	@dutils.save_model_epochs(filename = saved_state_filename, epochs = 10)
	def iteration(epoch = 1, model = model, optimizer = optimizer):
		# Forward pass: Compute predicted y by passing x to the model
		y_pred = model(x)
		# Compute and print loss
		loss = criterion(y_pred, y)
		dutils.l2regularization(model, loss) # Manually adding a regularizer
		print('Epoch:', epoch, '|  Loss:', loss.item())

		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	epoch = 1
	while epoch <= 1000:
		epoch = iteration(epoch = epoch)

	print('Finished Training')
	print('Training Accuracy: %s\nTest Accuracy: %s' % (getAccuracy(model, x, y), getAccuracy(model, x_test, y_test)))


def getAccuracy(model, x, y):
	correct = 0
	total = 0
	with torch.no_grad():
		outputs = model(x)
		_, predicted = torch.max(outputs, 1)
		_, actual = torch.max(y, 1)
		print(outputs.shape, predicted.shape, y.shape, actual.shape)
		total += y.size(0)
		correct += (predicted == actual).sum().item()
	return 'Accuracy: %f %%' % (100 * correct / total)

	# plt.imshow(x[0])
	# plt.show()

if __name__ == '__main__':
	main()

