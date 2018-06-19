import torch.nn as nn
import math
from .util_models import ConcatTable, CaddTable, Identity
from opt import opt


def Residual(numIn, numOut, *arg, stride=1, net_type='preact', useConv=False, **kw):
    con = ConcatTable([convBlock(numIn, numOut, stride, net_type),
                       skipLayer(numIn, numOut, stride, useConv)])
    cadd = CaddTable(True)
    return nn.Sequential(con, cadd)


def convBlock(numIn, numOut, stride, net_type):
    s_list = []
    if net_type != 'no_preact':
        s_list.append(nn.BatchNorm2d(numIn))
        s_list.append(nn.ReLU(True))

    conv1 = nn.Conv2d(numIn, numOut // 2, kernel_size=1)
    if opt.init:
        nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
    s_list.append(conv1)

    s_list.append(nn.BatchNorm2d(numOut // 2))
    s_list.append(nn.ReLU(True))

    conv2 = nn.Conv2d(numOut // 2, numOut // 2, kernel_size=3, stride=stride, padding=1)
    if opt.init:
        nn.init.xavier_normal(conv2.weight)
    s_list.append(conv2)
    s_list.append(nn.BatchNorm2d(numOut // 2))
    s_list.append(nn.ReLU(True))

    conv3 = nn.Conv2d(numOut // 2, numOut, kernel_size=1)
    if opt.init:
        nn.init.xavier_normal(conv3.weight)
    s_list.append(conv3)

    return nn.Sequential(*s_list)


def skipLayer(numIn, numOut, stride, useConv):
    if numIn == numOut and stride == 1 and not useConv:
        return Identity()
    else:
        conv1 = nn.Conv2d(numIn, numOut, kernel_size=1, stride=stride)
        if opt.init:
            nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
        return nn.Sequential(
            nn.BatchNorm2d(numIn),
            nn.ReLU(True),
            conv1
        )
