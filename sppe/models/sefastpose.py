import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import initializer
from opt import opt
from sppe.models.layers.pixelshuffle import PixelShuffle
from sppe.models.layers.se_resnet import SEResnet
from sppe.models.layers.duc import DUC
from sppe.models.fn import _try_load_parameters, _load_from_pytorch


if opt.syncbn:
    norm_layer = mx.gluon.contrib.nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm


class FastPose_SE(HybridBlock):
    try_load_parameters = _try_load_parameters
    load_from_pytorch = _load_from_pytorch
    deconv_dim = 256

    def reload_base(self, ctx=mx.cpu()):
        if opt.use_pretrained_base:
            print('===Pretrain Base===')
            from gluoncv.model_zoo import get_model
            # self.preact.initialize(mx.init.MSRAPrelu(), ctx=ctx)
            base_network = get_model('resnet101_v1b', pretrained=True, root='../exp/pretrain', ctx=ctx)

            self.preact.try_load_parameters(model=base_network, ctx=ctx)

    def __init__(self, ctx=mx.cpu(), pretrained=True, **kwargs):
        super(FastPose_SE, self).__init__()

        self.preact = SEResnet('resnet101', norm_layer=norm_layer, **kwargs)
        self.reload_base()

        self.shuffle1 = PixelShuffle(2)
        self.duc1 = DUC(1024, upscale_factor=2, norm_layer=norm_layer, **kwargs)
        self.duc2 = DUC(512, upscale_factor=2, norm_layer=norm_layer, **kwargs)

        self.conv_out = nn.Conv2D(
            channels=opt.nClasses,
            kernel_size=3,
            strides=1,
            padding=1,
            weight_initializer=initializer.Normal(0.001),
            bias_initializer=initializer.Zero()
        )

    def hybrid_forward(self, F, x):
        x = self.preact(x)
        x = self.shuffle1(x)
        x = self.duc1(x)
        x = self.duc2(x)

        x = self.conv_out(x)
        return x
