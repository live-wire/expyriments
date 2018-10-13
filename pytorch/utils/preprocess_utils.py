import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from skimage import io, transform
from torch.autograd import Variable

import unicodedata
import string

import sys
sys.path.append('..')

from utils.flags import DEBUG_FLAG

def debug(*args, **kwargs):
    if (DEBUG_FLAG):
        print(*args,**kwargs)


def resize2d(img, size):
    return transform.resize(img, size)

def getCategoricalTensor(y, rows, cols):
	y_final = torch.zeros(rows, cols)
	for ind, val in enumerate(y):
		y_final[ind][val.int()] = 1
	return y_final

def splitDataset(X, y):
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.33, random_state=42)
	return X_train, X_test, y_train, y_test

def flattenRows(x):
	ret = []
	for i, item in enumerate(x):
		ret.append(item.flatten())
	return np.array(ret)

def shuffle_together(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor