import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

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


