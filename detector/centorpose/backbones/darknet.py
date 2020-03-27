import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

       
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        #self.layer4 = self._make_layer([256, 512], layers[3])
        #self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256]

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, planes, blocks):
        layers = []
        #  downsample
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        #  blocks
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.interpolate(x, size=(128, 128), 
            mode="bilinear", align_corners=True)

        return x


def darknet21(cfg,is_train=True, **kwargs):
    model = DarkNet([1, 1, 2, 2, 1])
    if is_train and cfg.BACKBONE.INIT_WEIGHTS:
        if isinstance(cfg.BACKBONE.PRETRAINED, str):
            model.load_state_dict(torch.load(cfg.BACKBONE.PRETRAINED))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(cfg.BACKBONE.PRETRAINED))
    return model

def darknet53(num_layers, cfg):
    model = DarkNet([1, 2, 8])
    #if is_train and cfg.BACKBONE.INIT_WEIGHTS:
    #    if isinstance(cfg.BACKBONE.PRETRAINED, str):
    #        model.load_state_dict(torch.load(cfg.BACKBONE.PRETRAINED))
    #    else:
    #        raise Exception("darknet request a pretrained path. got [{}]".format(cfg.BACKBONE.PRETRAINED))
    return model
