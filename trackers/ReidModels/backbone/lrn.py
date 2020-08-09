import torch
from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld
from torch.autograd import Function, Variable
from torch.nn import Module


# function interface, internal, do not use this one!!!
class SpatialCrossMapLRNFunc(Function):

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        self.save_for_backward(input)
        self.lrn = SpatialCrossMapLRNOld(self.size, self.alpha, self.beta, self.k)
        self.lrn.type(input.type())
        return self.lrn.forward(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return self.lrn.backward(input, grad_output)


# use this one instead
class SpatialCrossMapLRN(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return SpatialCrossMapLRNFunc(self.size, self.alpha, self.beta, self.k)(input)