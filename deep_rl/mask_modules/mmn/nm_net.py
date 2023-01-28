#-*- coding: utf-8 -*-
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import *

class NMNet(nn.Module):
    def __init__(self, weight_shape, z_dim):
        super(NMNet, self).__init__()
        self.original_shape = np.array(weight_shape)
        if len(self.original_shape) == 1:
            self.weight_shape = np.concatenate([np.ones(1, dtype=np.int64), self.original_shape])
        else:
            self.weight_shape = self.original_shape[::-1]
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, z_dim * int(self.weight_shape[0]))
        self.fc2 = nn.Linear(z_dim, int(self.weight_shape[1]))

    def forward(self, x):
        x = tensor(x)
        # TODO implement a technique if it exist to enforce sparse representations
        # I know ReLU activation enforces some form of sparsity as negative values are
        # zerod out, but I want to enforce a stricter sparsity prior
        x = F.relu(self.fc1(x))
        x = x.reshape(self.weight_shape[0], self.z_dim)
        x = F.relu(self.fc2(x))
        return x.reshape(*self.original_shape)


try:
    from nupic.torch.modules import KWinners
except:
    pass
class NMNetKWinners(nn.Module):
    def __init__(self, weight_shape, z_dim, percent_on=0.3):
        super(NMNetKWinners, self).__init__()
        self.original_shape = np.array(weight_shape)
        if len(self.original_shape) == 1:
            self.weight_shape = np.concatenate([np.ones(1, dtype=np.int64), self.original_shape])
        else:
            self.weight_shape = self.original_shape[::-1]
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, z_dim * int(self.weight_shape[0]))
        self.fc2 = nn.Linear(z_dim, int(self.weight_shape[1]))

        # NOTE fix me, the duty_cycle_period.
        self.act1 = KWinners(n=z_dim*int(self.weight_shape[0]), percent_on=percent_on, \
            boost_strength=1.0, duty_cycle_period=200)
        self.act2 = KWinners(n=int(self.weight_shape[1]), percent_on=percent_on, \
            boost_strength=1.0, duty_cycle_period=200)

    def forward(self, x):
        x = tensor(x)
        # TODO implement a technique if it exist to enforce sparse representations
        # I know ReLU activation enforces some form of sparsity as negative values are
        # zerod out, but I want to enforce a stricter sparsity prior
        x = self.act1(self.fc1(x))
        x = x.reshape(self.weight_shape[0], self.z_dim)
        #print(x.sum(), x.shape)
        x = self.fc2(x)
        if x.shape[0] < 5 and x.shape[1] < 5:
            # avoid kwinner activation
            x = F.relu(x)
        else:
            x = self.act2(x)
        #print(x.sum(), x.shape)
        return x.reshape(*self.original_shape)

def layer_init_nm(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.fc.weight.data)
    layer.fc.weight.data.mul_(w_scale)
    nn.init.constant_(layer.fc.bias.data, 0)

    nn.init.orthogonal_(layer.nm.fc1.weight.data)
    layer.nm.fc1.weight.data.mul_(w_scale)
    nn.init.constant_(layer.nm.fc1.bias.data, 0)

    nn.init.orthogonal_(layer.nm.fc2.weight.data)
    layer.nm.fc2.weight.data.mul_(w_scale)
    nn.init.constant_(layer.nm.fc2.bias.data, 0)
    return layer

def layer_init_nm_pnn(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.std.weight.data)
    layer.std.weight.data.mul_(w_scale)
    nn.init.constant_(layer.std.bias.data, 0)

    nn.init.orthogonal_(layer.in_nm.weight.data)
    layer.in_nm.weight.data.mul_(w_scale)
    nn.init.constant_(layer.in_nm.bias.data, 0)

    nn.init.orthogonal_(layer.out_nm.weight.data)
    layer.out_nm.weight.data.mul_(w_scale)
    nn.init.constant_(layer.out_nm.bias.data, 0)
    return layer

class LinearMask(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearMask, self).__init__(in_features, out_features, bias)

    def forward(self, x, mask):
        params = self.weight * mask
        return F.linear(x, params, self.bias)
