# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

from .dcn import DeformConv, ModulatedDeformConv
from .SE_module import SELayer


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 reduction=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if reduction:
            self.se = SELayer(planes)
        self.reduc = reduction

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, reduction=False,
                 norm_layer=nn.BatchNorm2d,
                 dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = dcn is not None

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
            self.with_modulated_dcn = dcn.get('MODULATED', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27

            self.conv2_offset = nn.Conv2d(
                planes,
                self.deformable_groups * offset_channels,
                kernel_size=3,
                stride=stride,
                padding=1)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                deformable_groups=self.deformable_groups,
                bias=False)

        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.with_dcn:
            out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = F.relu(self.bn2(self.conv2(out, offset, mask)))
        else:
            offset = self.conv2_offset(out)
            out = F.relu(self.bn2(self.conv2(out, offset)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class SEResnet(nn.Module):
    """ SEResnet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d,
                 dcn=None, stage_with_dcn=(False, False, False, False)):
        super(SEResnet, self).__init__()
        self._norm_layer = norm_layer
        assert architecture in ["resnet18", "resnet50", "resnet101", 'resnet152']
        layers = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
        }
        self.inplanes = 64
        if architecture == "resnet18" or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_dcn = [dcn if with_dcn else None for with_dcn in stage_with_dcn]

        self.layer1 = self.make_layer(self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion, momentum=0.1),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes,
                                stride, downsample, reduction=True,
                                norm_layer=self._norm_layer, dcn=dcn))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample,
                                norm_layer=self._norm_layer, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=self._norm_layer, dcn=dcn))

        return nn.Sequential(*layers)
