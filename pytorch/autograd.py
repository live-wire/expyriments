# Trying autograd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np


a = torch.ones(1,5, dtype=torch.float, requires_grad=True)

print(a.grad)
b = 2*a
c = b*b
d = c * 4

d = d.mean()

# dot = make_dot(c)
# dot.format = 'svg'
# dot.render()


d.backward()

print(a.grad)



