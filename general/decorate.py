# Custom decorators <3
import time

# Checks how long a function takes to execute
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, (time2-time1)))
        return ret
    return wrap

# Overrides the print function to make it print nothing
def noprint(f):
	def wrap(*args, **kwargs):
		global print
		temp = print
		print = dummyFunction
		ret = f(*args, **kwargs)
		print = temp
		return ret
	return wrap

def dummyFunction(*args, **kwargs):
	pass


