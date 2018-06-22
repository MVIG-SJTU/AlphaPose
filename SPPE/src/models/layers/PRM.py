import torch.nn as nn
from .util_models import ConcatTable, CaddTable, Identity
import math
from opt import opt


class Residual(nn.Module):
    def __init__(self, numIn, numOut, inputResH, inputResW, stride=1,
                 net_type='preact', useConv=False, baseWidth=9, cardinality=4):
        super(Residual, self).__init__()

        self.con = ConcatTable([convBlock(numIn, numOut, inputResH,
                                          inputResW, net_type, baseWidth, cardinality, stride),
                                skipLayer(numIn, numOut, stride, useConv)])
        self.cadd = CaddTable(True)

    def forward(self, x):
        out = self.con(x)
        out = self.cadd(out)
        return out


def convBlock(numIn, numOut, inputResH, inputResW, net_type, baseWidth, cardinality, stride):
    numIn = int(numIn)
    numOut = int(numOut)

    addTable = ConcatTable()
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

    conv2 = nn.Conv2d(numOut // 2, numOut // 2,
                      kernel_size=3, stride=stride, padding=1)
    if opt.init:
        nn.init.xavier_normal(conv2.weight)
    s_list.append(conv2)

    s = nn.Sequential(*s_list)
    addTable.add(s)

    D = math.floor(numOut // baseWidth)
    C = cardinality
    s_list = []

    if net_type != 'no_preact':
        s_list.append(nn.BatchNorm2d(numIn))
        s_list.append(nn.ReLU(True))

    conv1 = nn.Conv2d(numIn, D, kernel_size=1, stride=stride)
    if opt.init:
        nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / C))

    s_list.append(conv1)
    s_list.append(nn.BatchNorm2d(D))
    s_list.append(nn.ReLU(True))
    s_list.append(pyramid(D, C, inputResH, inputResW))
    s_list.append(nn.BatchNorm2d(D))
    s_list.append(nn.ReLU(True))

    a = nn.Conv2d(D, numOut // 2, kernel_size=1)
    a.nBranchIn = C
    if opt.init:
        nn.init.xavier_normal(a.weight, gain=math.sqrt(1 / C))
    s_list.append(a)

    s = nn.Sequential(*s_list)
    addTable.add(s)

    elewiswAdd = nn.Sequential(
        addTable,
        CaddTable(False)
    )
    conv2 = nn.Conv2d(numOut // 2, numOut, kernel_size=1)
    if opt.init:
        nn.init.xavier_normal(conv2.weight, gain=math.sqrt(1 / 2))
    model = nn.Sequential(
        elewiswAdd,
        nn.BatchNorm2d(numOut // 2),
        nn.ReLU(True),
        conv2
    )
    return model


def pyramid(D, C, inputResH, inputResW):
    pyraTable = ConcatTable()
    sc = math.pow(2, 1 / C)
    for i in range(C):
        scaled = 1 / math.pow(sc, i + 1)
        conv1 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1)
        if opt.init:
            nn.init.xavier_normal(conv1.weight)
        s = nn.Sequential(
            nn.FractionalMaxPool2d(2, output_ratio=(scaled, scaled)),
            conv1,
            nn.UpsamplingBilinear2d(size=(int(inputResH), int(inputResW))))
        pyraTable.add(s)
    pyra = nn.Sequential(
        pyraTable,
        CaddTable(False)
    )
    return pyra


class skipLayer(nn.Module):
    def __init__(self, numIn, numOut, stride, useConv):
        super(skipLayer, self).__init__()
        self.identity = False

        if numIn == numOut and stride == 1 and not useConv:
            self.identity = True
        else:
            conv1 = nn.Conv2d(numIn, numOut, kernel_size=1, stride=stride)
            if opt.init:
                nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
            self.m = nn.Sequential(
                nn.BatchNorm2d(numIn),
                nn.ReLU(True),
                conv1
            )

    def forward(self, x):
        if self.identity:
            return x
        else:
            return self.m(x)
