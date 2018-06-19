import torch.nn as nn
from .layers.PRM import Residual as ResidualPyramid
from .layers.Residual import Residual as Residual
from torch.autograd import Variable
import torch
from opt import opt
import math


class Hourglass(nn.Module):
    def __init__(self, n, nFeats, nModules, inputResH, inputResW, net_type, B, C):
        super(Hourglass, self).__init__()

        self.ResidualUp = ResidualPyramid if n >= 2 else Residual
        self.ResidualDown = ResidualPyramid if n >= 3 else Residual

        self.depth = n
        self.nModules = nModules
        self.nFeats = nFeats
        self.net_type = net_type
        self.B = B
        self.C = C
        self.inputResH = inputResH
        self.inputResW = inputResW

        up1 = self._make_residual(self.ResidualUp, False, inputResH, inputResW)
        low1 = nn.Sequential(
            nn.MaxPool2d(2),
            self._make_residual(self.ResidualDown, False, inputResH / 2, inputResW / 2)
        )
        if n > 1:
            low2 = Hourglass(n - 1, nFeats, nModules, inputResH / 2, inputResW / 2, net_type, B, C)
        else:
            low2 = self._make_residual(self.ResidualDown, False, inputResH / 2, inputResW / 2)

        low3 = self._make_residual(self.ResidualDown, True, inputResH / 2, inputResW / 2)
        up2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.upperBranch = up1
        self.lowerBranch = nn.Sequential(
            low1,
            low2,
            low3,
            up2
        )

    def _make_residual(self, resBlock, useConv, inputResH, inputResW):
        layer_list = []
        for i in range(self.nModules):
            layer_list.append(resBlock(self.nFeats, self.nFeats, inputResH, inputResW,
                                       stride=1, net_type=self.net_type, useConv=useConv,
                                       baseWidth=self.B, cardinality=self.C))
        return nn.Sequential(*layer_list)

    def forward(self, x: Variable):
        up1 = self.upperBranch(x)
        up2 = self.lowerBranch(x)
        # out = up1 + up2
        out = torch.add(up1, up2)
        return out


class PyraNet(nn.Module):
    def __init__(self):
        super(PyraNet, self).__init__()

        B, C = opt.baseWidth, opt.cardinality
        self.inputResH = opt.inputResH / 4
        self.inputResW = opt.inputResW / 4
        self.nStack = opt.nStack

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        if opt.init:
            nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 3))

        cnv1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        r1 = nn.Sequential(
            ResidualPyramid(64, 128, opt.inputResH / 2, opt.inputResW / 2,
                            stride=1, net_type='no_preact', useConv=False, baseWidth=B, cardinality=C),
            nn.MaxPool2d(2)
        )
        r4 = ResidualPyramid(128, 128, self.inputResH, self.inputResW,
                             stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        r5 = ResidualPyramid(128, opt.nFeats, self.inputResH, self.inputResW,
                             stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        self.preact = nn.Sequential(
            cnv1,
            r1,
            r4,
            r5
        )

        self.stack_lin = nn.ModuleList()
        self.stack_out = nn.ModuleList()
        self.stack_lin_ = nn.ModuleList()
        self.stack_out_ = nn.ModuleList()

        for i in range(self.nStack):
            hg = Hourglass(4, opt.nFeats, opt.nResidual, self.inputResH, self.inputResW, 'preact', B, C)
            conv1 = nn.Conv2d(opt.nFeats, opt.nFeats, kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
            lin = nn.Sequential(
                hg,
                nn.BatchNorm2d(opt.nFeats),
                nn.ReLU(True),
                conv1,
                nn.BatchNorm2d(opt.nFeats),
                nn.ReLU(True)
            )
            tmpOut = nn.Conv2d(opt.nFeats, opt.nClasses, kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(tmpOut.weight)
            self.stack_lin.append(lin)
            self.stack_out.append(tmpOut)
            if i < self.nStack - 1:
                lin_ = nn.Conv2d(opt.nFeats, opt.nFeats, kernel_size=1, stride=1, padding=0)
                tmpOut_ = nn.Conv2d(opt.nClasses, opt.nFeats, kernel_size=1, stride=1, padding=0)
                if opt.init:
                    nn.init.xavier_normal(lin_.weight)
                    nn.init.xavier_normal(tmpOut_.weight)
                self.stack_lin_.append(lin_)
                self.stack_out_.append(tmpOut_)

    def forward(self, x: Variable):
        out = []
        inter = self.preact(x)
        for i in range(self.nStack):
            lin = self.stack_lin[i](inter)
            tmpOut = self.stack_out[i](lin)
            out.append(tmpOut)
            if i < self.nStack - 1:
                lin_ = self.stack_lin_[i](lin)
                tmpOut_ = self.stack_out_[i](tmpOut)
                inter = inter + lin_ + tmpOut_
        return out


class PyraNet_Inference(nn.Module):
    def __init__(self):
        super(PyraNet_Inference, self).__init__()

        B, C = opt.baseWidth, opt.cardinality
        self.inputResH = opt.inputResH / 4
        self.inputResW = opt.inputResW / 4
        self.nStack = opt.nStack

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        if opt.init:
            nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 3))

        cnv1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        r1 = nn.Sequential(
            ResidualPyramid(64, 128, opt.inputResH / 2, opt.inputResW / 2,
                            stride=1, net_type='no_preact', useConv=False, baseWidth=B, cardinality=C),
            nn.MaxPool2d(2)
        )
        r4 = ResidualPyramid(128, 128, self.inputResH, self.inputResW,
                             stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        r5 = ResidualPyramid(128, opt.nFeats, self.inputResH, self.inputResW,
                             stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        self.preact = nn.Sequential(
            cnv1,
            r1,
            r4,
            r5
        )

        self.stack_lin = nn.ModuleList()
        self.stack_out = nn.ModuleList()
        self.stack_lin_ = nn.ModuleList()
        self.stack_out_ = nn.ModuleList()

        for i in range(self.nStack):
            hg = Hourglass(4, opt.nFeats, opt.nResidual,
                           self.inputResH, self.inputResW, 'preact', B, C)
            conv1 = nn.Conv2d(opt.nFeats, opt.nFeats,
                              kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
            lin = nn.Sequential(
                hg,
                nn.BatchNorm2d(opt.nFeats),
                nn.ReLU(True),
                conv1,
                nn.BatchNorm2d(opt.nFeats),
                nn.ReLU(True)
            )
            tmpOut = nn.Conv2d(opt.nFeats, opt.nClasses,
                               kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(tmpOut.weight)
            self.stack_lin.append(lin)
            self.stack_out.append(tmpOut)
            if i < self.nStack - 1:
                lin_ = nn.Conv2d(opt.nFeats, opt.nFeats,
                                 kernel_size=1, stride=1, padding=0)
                tmpOut_ = nn.Conv2d(opt.nClasses, opt.nFeats,
                                    kernel_size=1, stride=1, padding=0)
                if opt.init:
                    nn.init.xavier_normal(lin_.weight)
                    nn.init.xavier_normal(tmpOut_.weight)
                self.stack_lin_.append(lin_)
                self.stack_out_.append(tmpOut_)

    def forward(self, x: Variable):
        inter = self.preact(x)
        for i in range(self.nStack):
            lin = self.stack_lin[i](inter)
            tmpOut = self.stack_out[i](lin)
            out = tmpOut
            if i < self.nStack - 1:
                lin_ = self.stack_lin_[i](lin)
                tmpOut_ = self.stack_out_[i](tmpOut)
                inter = inter + lin_ + tmpOut_
        return out


def createModel(**kw):
    model = PyraNet()
    return model


def createModel_Inference(**kw):
    model = PyraNet_Inference()
    return model
