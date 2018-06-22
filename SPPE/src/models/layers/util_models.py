import torch
import torch.nn as nn
from torch.autograd import Variable


class ConcatTable(nn.Module):
    def __init__(self, module_list=None):
        super(ConcatTable, self).__init__()

        self.modules_list = nn.ModuleList(module_list)

    def forward(self, x: Variable):
        y = []
        for i in range(len(self.modules_list)):
            y.append(self.modules_list[i](x))
        return y

    def add(self, module):
        self.modules_list.append(module)


class CaddTable(nn.Module):
    def __init__(self, inplace=False):
        super(CaddTable, self).__init__()
        self.inplace = inplace

    def forward(self, x: Variable or list):
        return torch.stack(x, 0).sum(0)


class Identity(nn.Module):
    def __init__(self, params=None):
        super(Identity, self).__init__()
        self.params = nn.ParameterList(params)

    def forward(self, x: Variable or list):
        return x
