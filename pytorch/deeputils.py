import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import inspect

# Model Save and Load Epoch-State
load_model = True

def save_model_epochs(filename = 'intermediate/backup.pt', epochs = 10, debug = False):
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
					kwargs['epoch'] = kwargs['epoch'] + 1

			# Calling the actual iteration function
			fn(*args, **kwargs)

			# After
			load_model = False
			state = {
			'epoch': kwargs['epoch'],
			'state_dict': kwargs['model'].state_dict(),
			'optimizer': kwargs['optimizer'].state_dict(),
			}
			if kwargs['epoch'] % epochs == 0:
				torch.save(state, filename)

			# returning state
			return state['epoch'] + 1


		return wrapper
	return wrapper_outer

