import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from deep_utils import save_model_epochs

class DynamicNet(torch.nn.Module):
	def __init__(self, D_in, D_H, N_H, D_out):
		super(DynamicNet, self).__init__()
		self.hidden_layers = N_H
		self.inputLinear = torch.nn.Linear(D_in, D_H)
		self.middleLinear = torch.nn.Linear(D_H, D_H)
		self.outputLinear = torch.nn.Linear(D_H, D_out)

	def forward(self, x):
		h_activation = self.inputLinear(x).clamp(min=0)
		h_activation_test = F.relu(self.inputLinear(x))
		for _ in range(self.hidden_layers):
			h_activation = self.middleLinear(h_activation).clamp(min=0)
		y_predicted = self.outputLinear(h_activation)
		return y_predicted 


def main():	
	# Fetching Data and Preprocessing
	from datagen import Spiral
	resume_execution = True
	data = Spiral()
	x, y = data.generate()
	print(x.shape, y.shape)

	# N is batch size; D_in is input dimension;
	# H is hidden dimension; D_out is output dimension.
	N, D_in, D_H, N_H, D_out = 300, 2, 500, 1, 3

	saved_state_filename = 'intermediate/%d_%d_%d_%d_DynamicNet.pt' % (D_in, D_H, N_H, D_out)


	# Tensors for the dataset
	x = torch.tensor(x).float()
	y = torch.tensor(y).long()
	y_final = torch.zeros(N, D_out)
	for ind, val in enumerate(y):
		y_final[ind][val.int()] = 1


	# Construct our model by instantiating the class defined above
	model = DynamicNet(D_in, D_H, N_H, D_out)
	# Construct our loss function and an Optimizer. Training this strange model with
	# vanilla stochastic gradient descent is tough, so we use momentum

	# criterion = torch.nn.MSELoss(reduction='sum')
	criterion = torch.nn.CrossEntropyLoss()
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


	epoch = 1
	while epoch <= 500:

		@save_model_epochs(filename = saved_state_filename, epochs = 10)
		def iteration(epoch = epoch, model = model, optimizer = optimizer):
			# Forward pass: Compute predicted y by passing x to the model
			y_pred = model(x)
			# Compute and print loss
			loss = criterion(y_pred, y_final)
			print('Epoch:', epoch, '|  Loss:', loss.item())

			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		epoch = iteration()

	print('Finished Training')

	correct = 0
	total = 0
	with torch.no_grad():
		outputs = model(x)
		_, predicted = torch.max(outputs.data, 1)
		# print(outputs, predicted)
		total += y_final.size(0)
		correct += (predicted == y).sum().item()

	print('Training Accuracy: %f %%' % (100 * correct / total))

if __name__ == '__main__':
	main()

