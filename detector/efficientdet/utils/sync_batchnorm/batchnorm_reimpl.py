#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : batchnorm_reimpl.py
# Author : acgtyrant
# Date   : 11/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['BatchNorm2dReimpl']


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean

        self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.detach()
        )
        unbias_var = sumvar / (numel - 1)
        self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * unbias_var.detach()
        )

        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (
                (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1) *
                self.weight.unsqueeze(1) + self.bias.unsqueeze(1))

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()

