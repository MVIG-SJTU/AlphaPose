from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from sppe.models.layers.pixelshuffle import PixelShuffle


class DUC(HybridBlock):
    def __init__(self, planes, upscale_factor=2, norm_layer=nn.BatchNorm, **kwargs):
        super(DUC, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(planes, kernel_size=3, padding=1, use_bias=False)
            self.bn = norm_layer(**kwargs)
            self.relu = nn.Activation('relu')
            self.pixel_shuffle = PixelShuffle(upscale_factor)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)

        return x
