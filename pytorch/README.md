# Pytorch playground :fire:

Official tutorials [here](https://pytorch.org/tutorials/)



## Utilities implemented

- Saving Model Epochs (decorator) see `utils/deep_utils.py`.
	- Send `visualization=True` in the decorator arguments to enable saving visualization data
	- Use this decorator like the example below:
	```
	import utilities.deep_utils as dutils
	...
	# Filename to store model states (use this format with 'intermediate' to make use of visualization utilities)
	saved_state_filename = 'intermediate/TenNetVis.pt'
	# preprocessing and preparing dataset
	data = x, y, x_test, y_test
	# Initializing model, criterion and optimizer
	...
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
	```

- Visualization Utilities see `utils/vis_utils.py`
	- Train, Test accuracy % vs epochs
	- Train Loss vs Epochs
	- Run visualization utilities in a separate window like:
	```
	python utils/vis_utils.py --net tennet
	```
	Here ^ replace tennet with your net implementation. 
	> It assumes your visualization data is being saved in the "tennet" folder inside `intermediate/visualization_data.pt`
