import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np
from SPPE.src.utils.img import flip_v, shuffleLR
from SPPE.src.utils.eval import getPrediction
from SPPE.src.models.hgPRM import createModel_Inference

import visdom
import time


import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


# batch_size = 128 * 8


def gaussian(size):
    '''
    Generate a 2D gaussian array
    '''
    sigma = 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g = g[np.newaxis, :]
    return g


class InferenNet(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet, self).__init__()

        model = createModel_Inference().cuda()
        print('Loading pose model from {}'.format('./models/sppe/pyra_4.pth'))
        model.load_state_dict(torch.load('./models/sppe/pyra_4.pth'))
        model.eval()
        self.pyranet = model
        self.gaussian = nn.Conv2d(17, 17, kernel_size=kernel_size,
                                  stride=1, padding=2, groups=17, bias=False)

        g = torch.from_numpy(gaussian(kernel_size)).clone()
        g = torch.unsqueeze(g, 1)
        g = g.repeat(17, 1, 1, 1)
        assert g.shape == self.gaussian.weight.data.shape
        self.gaussian.weight.data = g.float()

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        flip_out = self.pyranet(flip_v(x))
        flip_out = flip_out.narrow(1, 0, 17)

        flip_out = flip_v(shuffleLR(
            flip_out, self.dataset))

        out = (flip_out + out) / 2
        out = self.gaussian(F.relu(out, inplace=True))

        return out


class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet_fast, self).__init__()

        model = createModel_Inference().cuda()
        print('Loading pose model from {}'.format('./models/sppe/pyra_4.pth'))
        model.load_state_dict(torch.load('./models/sppe/pyra_4.pth'))
        model.eval()
        self.pyranet = model
        self.gaussian = nn.Conv2d(17, 17, kernel_size=kernel_size,
                                  stride=1, padding=2, groups=17, bias=False)

        g = torch.from_numpy(gaussian(kernel_size)).clone()
        g = torch.unsqueeze(g, 1)
        g = g.repeat(17, 1, 1, 1)
        assert g.shape == self.gaussian.weight.data.shape
        self.gaussian.weight.data = g.float()

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        out = self.gaussian(F.relu(out, inplace=True))

        return out
