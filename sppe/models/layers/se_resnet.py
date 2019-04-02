import mxnet.gluon.nn as nn
from mxnet.gluon.block import HybridBlock

from sppe.models.layers.se_module import SELayer
from sppe.models.fn import _try_load_parameters


class Bottleneck(HybridBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False, norm_layer=nn.BatchNorm, **kwargs):
        super(Bottleneck, self).__init__()

        with self.name_scope():
            self.conv1 = nn.Conv2D(planes, kernel_size=1, use_bias=False)
            self.bn1 = norm_layer(**kwargs)
            self.conv2 = nn.Conv2D(planes, kernel_size=3, strides=stride, padding=1, use_bias=False)
            self.bn2 = norm_layer(**kwargs)
            self.conv3 = nn.Conv2D(planes * 4, kernel_size=1, use_bias=False)
            self.bn3 = norm_layer(**kwargs)

        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def hybrid_forward(self, F, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out
        out = F.relu(out)

        return out


class SEResnet(HybridBlock):
    """ SEResnet """
    try_load_parameters = _try_load_parameters

    def __init__(self, architecture, norm_layer=nn.BatchNorm, **kwargs):
        super(SEResnet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.norm_layer = norm_layer
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck

        self.conv1 = nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False)
        self.bn1 = self.norm_layer(**kwargs)
        self.relu = nn.Activation('relu')
        self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

        self.layer1 = self.make_layer(self.block, 64, self.layers[0], **kwargs)
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2, **kwargs)
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2, **kwargs)

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2, **kwargs)

    def hybrid_forward(self, F, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential()
            downsample.add(nn.Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False))
            downsample.add(self.norm_layer(**kwargs))

        layers = nn.HybridSequential()
        if downsample is not None:
            layers.add(block(self.inplanes, planes, stride, downsample, reduction=True, norm_layer=self.norm_layer, **kwargs))
        else:
            layers.add(block(self.inplanes, planes, stride, downsample, norm_layer=self.norm_layer, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.add(block(self.inplanes, planes, norm_layer=self.norm_layer, **kwargs))

        return layers
