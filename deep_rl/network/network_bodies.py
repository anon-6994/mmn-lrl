#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class CTgraphConvBody(nn.Module):
    def __init__(self, in_channels=1):
        super(CTgraphConvBody, self).__init__()
        self.feature_dim = 16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1))
        self.conv2 = layer_init(nn.Conv2d(4, 8, kernel_size=3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(4 * 4 * 16, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class MNISTConvBody(nn.Module):
    def __init__(self, in_channels=1, noisy_linear=False):
        super(MNISTConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        #self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(6 * 6 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(6 * 6 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        #y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class FCBody_CL(nn.Module): # fcbody for continual learning setup
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_CL, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for layer in self.layers:
                x = self.gate(layer(x))
        return x, ret_act

from ..mask_modules.mmn.mask_nets import MultitaskMaskLinear
from ..mask_modules.mmn.mask_nets import NEW_MASK_RANDOM
from ..mask_modules.mmn.mask_nets import NEW_MASK_LINEAR_COMB
class FCBody_SS(nn.Module): # fcbody for supermask superposition continual learning algorithm
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM):
        super(FCBody_SS, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([MultitaskMaskLinear(dim_in, dim_out, discrete=discrete_mask, \
            num_tasks=num_tasks, new_mask_type=new_task_mask) \
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for layer in self.layers:
                x = self.gate(layer(x))
        return x, ret_act

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class DummyBody_CL(nn.Module):
    def __init__(self, state_dim, task_label_dim=None):
        super(DummyBody_CL, self).__init__()
        self.feature_dim = state_dim + (0 if task_label_dim is None else task_label_dim)
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        return x, []

class DummyBody_CL_Mask(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody_CL_Mask, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix='', mask=None):
        return x, []
