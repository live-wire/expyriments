import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import inspect

# Model Save and Load Epoch-State
load_model = True
from graphviz import Digraph
from torchviz import make_dot

import sys
sys.path.append('..')

from utils.vis_utils import Visualize
visualize = Visualize()

visualization_params_default = {
	'accuracy_vs_epoch': True,
	'loss_vs_epoch': True,
	'testAccuracy_vs_trainAccuracy': True,
	'learningRate_vs_epoch': True
}
def save_model_epochs(
	filename = 'intermediate/backup.pt',
	visualization = False,
	visualization_params = visualization_params_default,
	epochs = 10,
	save_epoch_states = 100,
	data = None,
	debug = False):

	def wrapper_outer(fn):
		# Decorator needs to see default values of the function
		argspec = inspect.getargspec(fn)
		defaultArguments = list(reversed(list(zip(reversed(argspec.args), reversed(argspec.defaults)))))
		def wrapper(*args, **kwargs):
			global load_model
			# all_kwargs = kwargs.copy()
			for arg, value in defaultArguments:
				if arg not in kwargs:
					kwargs[arg] = value
			if (debug):
				print('%s called with positional args %s and keyword args %s' % (fn.__name__, args, kwargs))

			# Before
			if os.path.isfile(filename) and load_model:
				state = torch.load(filename)
				if state['epoch']:
					print('Reloading from saved state - epoch:', state['epoch'])
					kwargs['epoch'] = state['epoch']
					kwargs['model'].load_state_dict(state['state_dict'])
					kwargs['optimizer'].load_state_dict(state['optimizer'])
					kwargs['criterion'].load_state_dict(state['criterion'])
					kwargs['epoch'] = kwargs['epoch'] + 1

			# Calling the actual iteration function
			fn(*args, **kwargs)

			# After
			load_model = False
			state = {
			'epoch': kwargs['epoch'],
			'state_dict': kwargs['model'].state_dict(),
			'optimizer': kwargs['optimizer'].state_dict(),
			'criterion': kwargs['criterion'].state_dict()
			}

			if kwargs['epoch'] % epochs == 0:
				saveModelState(state, filename)
				if visualization:
					displayVisualizations(data, kwargs['model'], kwargs['epoch'], kwargs['criterion'], kwargs['optimizer'], visualization_params)
			if kwargs['epoch'] % save_epoch_states == 0:
				saveModelState(state, filename + '_' + 'epoch_' + str(kwargs['epoch']))
			# returning state
			return state['epoch'] + 1
		return wrapper
	return wrapper_outer

def saveModelState(state, filename):
	try:
		torch.save(state, filename)
	except:
		os.makedirs(filename[:filename.rfind('/')])
		torch.save(state, filename)

def l2regularization(model, loss):
	lambda_ = torch.tensor(1.)
	l2_reg = torch.tensor(0.)
	for param in model.parameters():
	    l2_reg += torch.norm(param)
	loss += lambda_ * l2_reg

def showModel(variable):
	make_dot(variable).view()

def displayVisualizations(data, model, epoch, criterion, optimizer, visualization_params):
	global visualize
	if (visualization_params['accuracy_vs_epoch']):
		visualize.accuracyVsEpochs(data, model, epoch)
	if (visualization_params['loss_vs_epoch']):
		visualize.lossVsEpochs(data, model, epoch, criterion)
	if (visualization_params['learningRate_vs_epoch']):
		visualize.learningRateVsEpochs(optimizer, epoch)

	# visualize.showPlots()

def getLearningRate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr


def adjustLearningRate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def getAccuracy(model, x, y):
	oldstate = model.training
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		outputs = model(x)
		_, predicted = torch.max(outputs, 1)
		_, actual = torch.max(y, 1)
		print(outputs.shape, predicted.shape, y.shape, actual.shape)
		total += y.size(0)
		correct += (predicted == actual).sum().item()
	model.train(oldstate)
	return (100 * correct / total)

def getLoss(model, criterion, x, y):
	loss = torch.tensor(0)
	with torch.no_grad():
		y_pred = model(x)
		loss = criterion(y_pred, y)
	return loss.item()

# plot learning rate vs epoch
