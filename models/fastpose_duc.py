# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn

from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.SE_Resnet import SEResnet
from .layers.ShuffleResnet import ShuffleResnet


@SPPE.register_module
class FastPose_DUC(nn.Module):
    conv_dim = 256

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose_DUC, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if cfg['BACKBONE'] == 'shuffle':
            print('Load shuffle backbone...')
            backbone = ShuffleResnet
        elif cfg['BACKBONE'] == 'se-resnet':
            print('Load SE Resnet...')
            backbone = SEResnet
        else:
            print('Load Resnet...')
            backbone = ResNet

        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = backbone(
                f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = backbone(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm   # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.norm_layer = norm_layer

        stage1_cfg = cfg['STAGE1']
        stage2_cfg = cfg['STAGE2']
        stage3_cfg = cfg['STAGE3']

        self.duc1 = self._make_duc_stage(stage1_cfg, 2048, 1024)
        self.duc2 = self._make_duc_stage(stage2_cfg, 1024, 512)
        self.duc3 = self._make_duc_stage(stage3_cfg, 512, self.conv_dim)

        self.conv_out = nn.Conv2d(
            self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.duc3(out)

        out = self.conv_out(out)
        return out

    def _make_duc_stage(self, layer_config, inplanes, outplanes):
        layers = []

        shuffle = nn.PixelShuffle(2)
        inplanes //= 4
        layers.append(shuffle)
        for i in range(layer_config.NUM_CONV - 1):
            conv = nn.Conv2d(inplanes, inplanes, kernel_size=3,
                             padding=1, bias=False)
            norm_layer = self.norm_layer(inplanes, momentum=0.1)
            relu = nn.ReLU(inplace=True)
            layers += [conv, norm_layer, relu]
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                         padding=1, bias=False)
        norm_layer = self.norm_layer(outplanes, momentum=0.1)
        relu = nn.ReLU(inplace=True)
        layers += [conv, norm_layer, relu]
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
