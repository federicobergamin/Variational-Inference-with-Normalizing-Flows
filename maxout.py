import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Function


# class Maxout(nn.Module):
#     def __init__(self, pool_size):
#         super().__init__()
#         self._pool_size = pool_size
#
#     def forward(self, x):
#         assert x.shape[-1] % self._pool_size == 0, \
#             'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
#         m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
#         return m

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
        return m