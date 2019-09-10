import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from VAE_with_normalizing_flows.flows import PlanarFlow, NormalizingFlows
from VAE_with_normalizing_flows.maxout import Maxout
import numpy as np


# a = torch.arange(1*400).view(1,400)
# print(a)
# b = torch.Tensor([1,2,3,4,5,6]).unsqueeze(0)
# print(b)
# c = b.squeeze()
# print(c)
#
# print(Maxout(4)(a))
# print(Maxout(4)(a).shape)




a = torch.arange(4*3*5).view(4,3,5)
print(a)
print(a.shape)
print('----')
print(a.unsqueeze(0))
print(a.unsqueeze(0).shape)
print('----')

print(a.unsqueeze(1))

print(a.unsqueeze(1).shape)
print('----')

print(a.unsqueeze(2))
print(a.unsqueeze(2).shape)