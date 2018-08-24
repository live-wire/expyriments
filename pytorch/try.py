import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from deeputils import save_model_epochs

@save_model_epochs(debug = True)
def powpow(leng, name = 'Dhruv', model = {'pow'}):
	print('powpow', leng, name, model)

powpow(2, name='Sanjeev')

powpow(4)