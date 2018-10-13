# Predict nationality based on name

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Adding parent folder to path (Python3 workaround)
import sys
sys.path.append('..')

import utils.deep_utils as dutils
import utils.preprocess_utils as prep
import prepare

import unicodedata
import string

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--predict', help='Use the pretrained model to predict nationality')
parser.add_argument('--n', help='Number of predictions')
args = parser.parse_args()


torch.backends.cudnn.deterministic = True
torch.manual_seed(1973)

class NameNet(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(NameNet, self).__init__()

		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


def categoryFromOutput(all_categories, output):
	top_n, top_i = output.topk(1)
	category_i = top_i[0].item()
	return all_categories[category_i], category_i

def evaluate(rnn, line_tensor):
	hidden = rnn.initHidden()
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)
	return output

def predict(rnn, input_line, all_categories, n_predictions=3):
	print('\n> %s' % input_line)
	with torch.no_grad():
		output = evaluate(rnn ,prep.lineToTensor(input_line))

		# Get top N categories
		topv, topi = output.topk(n_predictions, 1, True)
		predictions = []

		for i in range(n_predictions):
			value = topv[0][i].item()
			category_index = topi[0][i].item()
			print('(%.2f) %s' % (value, all_categories[category_index]))
			predictions.append([value, all_categories[category_index]])

def main():
	all_letters = string.ascii_letters + " .,;'"
	n_letters = len(all_letters)
	category_lines, all_categories, n_categories = prepare.fetchData()
	print('Categories: ', all_categories, n_categories)
	torch.save((all_categories, n_categories), 'categories.pt');
	n_hidden = 128
	rnn = NameNet(n_letters, n_hidden, n_categories)

	input = prep.lineToTensor('Albert')
	hidden = torch.zeros(1, n_hidden)

	output, next_hidden = rnn(input[0], hidden)
	print(output)

	criterion = nn.NLLLoss()
	learning_rate = 0.005

	learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

	def train(category_tensor, line_tensor):
		hidden = rnn.initHidden()

		rnn.zero_grad()

		for i in range(line_tensor.size()[0]):
			output, hidden = rnn(line_tensor[i], hidden)

		loss = criterion(output, category_tensor)
		loss.backward()

		# Add parameters' gradients to their values, multiplied by learning rate
		for p in rnn.parameters():
			p.data.add_(-learning_rate, p.grad.data)

		return output, loss.item()

	import random

	def randomChoice(l):
		return l[random.randint(0, len(l) - 1)]

	def randomTrainingExample():
		category = randomChoice(all_categories)
		line = randomChoice(category_lines[category])
		category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
		line_tensor = prep.lineToTensor(line)
		return category, line, category_tensor, line_tensor

	for i in range(10):
		category, line, category_tensor, line_tensor = randomTrainingExample()
		print('category =', category, '/ line =', line)

	import time
	import math

	n_iters = 100000
	print_every = 5000
	plot_every = 1000



	# Keep track of losses for plotting
	current_loss = 0
	all_losses = []

	def timeSince(since):
		now = time.time()
		s = now - since
		m = math.floor(s / 60)
		s -= m * 60
		return '%dm %ds' % (m, s)

	start = time.time()

	for iter in range(1, n_iters + 1):
		category, line, category_tensor, line_tensor = randomTrainingExample()
		output, loss = train(category_tensor, line_tensor)
		current_loss += loss

		# Print iter number, loss, name and guess
		if iter % print_every == 0:
			guess, guess_i = categoryFromOutput(all_categories, output)
			correct = '✓' if guess == category else '✗ (%s)' % category
			print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

	predict(rnn, 'Dovesky', all_categories)
	predict(rnn, 'Jackson', all_categories)
	predict(rnn, 'Satoshi', all_categories)

	dutils.saveModel(rnn, criterion, criterion)

def performPrediction(input_line, n_predictions=3):
	if args.n:
		n_predictions = args.n
	state = torch.load('model.pt')['state_dict']
	all_categories, n_categories = torch.load('categories.pt')
	print('Categories--', all_categories)
	all_letters = string.ascii_letters + " .,;'"
	n_letters = len(all_letters)
	n_hidden = 128
	rnn = NameNet(n_letters, n_hidden, n_categories)
	rnn.load_state_dict(state)
	rnn.eval()
	predict(rnn, input_line, all_categories)



if __name__ == '__main__':
	if (args.predict):
		performPrediction(args.predict)
	else:
		main()

