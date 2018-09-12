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
import utils.deep_utils as dutils
import utils.preprocess_utils as prep

torch.backends.cudnn.deterministic = True
torch.manual_seed(1973)

class TenNet(torch.nn.Module):
	def __init__(self):
		super(TenNet, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
		# 61470 + 6 + 16 + 120 + 84 + 10


	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.dropout(x, p=0, training=self.training)

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features



def main():
	x = np.load('data/X.npy')
	print(x.shape)
	# x = prep.flattenRows(x)
	print(x.shape)
	x = np.array(list(map(lambda a: np.reshape(prep.resize2d(a, (32, 32)), (1, 32, 32)), x)))
	print(x.shape)
	y = np.load('data/Y.npy')
	print(y.shape)
	x_train, x_test, y_train, y_test = prep.splitDataset(x,y)
	print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

	# N, D_in, D_H, N_H, D_out = 1381, 4096, 200, 4, 10

	saved_state_filename = 'intermediate/TenNetVis.pt'


	# Tensors for the dataset
	x = torch.tensor(x_train).float()
	y = torch.tensor(y_train).float()
	x_test = torch.tensor(x_test).float()
	y_test = torch.tensor(y_test).float()

	# data stores x_train, y_train, x_test, y_test
	data = x, y, x_test, y_test


	# Construct our model by instantiating the class defined above
	model = TenNet()
	model.train()
	# Construct our loss function and an Optimizer. Training this strange model with
	# vanilla stochastic gradient descent is tough, so we use momentum

	criterion = torch.nn.MSELoss(reduction='sum')
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


	@dutils.save_model_epochs(filename = saved_state_filename, epochs = 10, save_epoch_states = 100, data = data, visualization = True)
	def iteration(epoch = 1, model = model, optimizer = optimizer, data = data, criterion = criterion):
		# Forward pass: Compute predicted y by passing x to the model
		y_pred = model(x)
		# Compute and print loss
		loss = criterion(y_pred, y)
		# dutils.showModel(loss)
		# dutils.l2regularization(model, loss) # Manually adding a regularizer
		print('Epoch:', epoch, '|  Loss:', loss.item())

		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	epoch = 1
	while epoch <= 2000:
		epoch = iteration(epoch = epoch)

	print('Finished Training')
	model.eval()

	print('Training Accuracy: %s\nTest Accuracy: %s' % (getAccuracy(model, x, y), getAccuracy(model, x_test, y_test)))
	torch.save(model.state_dict(), 'model.pt')


def getAccuracy(model, x, y):
	correct = 0
	total = 0
	with torch.no_grad():
		outputs = model(x)
		_, predicted = torch.max(outputs, 1)
		_, actual = torch.max(y, 1)
		# print(outputs.shape, predicted.shape, y.shape, actual.shape)
		total += y.size(0)
		correct += (predicted == actual).sum().item()
	return 'Accuracy: %f %%' % (100 * correct / total)

	# plt.imshow(x[0])
	# plt.show()

if __name__ == '__main__':
	main()

