# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch
import torch.nn as nn
from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.SE_Resnet import SEResnet
from .layers.ShuffleResnet import ShuffleResnet

@SPPE.register_module
class FastPose_DUC_Dense(nn.Module):
    conv_dim = 256

    def __init__(self,norm_layer=nn.BatchNorm2d,**cfg):
        super(FastPose_DUC_Dense, self).__init__()
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

        # Init Backbone
        for m in self.preact.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)

        # Imagenet pretrain model
        import torchvision.models as tm
        if cfg['NUM_LAYERS'] == 152:
            ''' Load pretrained model '''
            x = tm.resnet152(pretrained=True)
        elif cfg['NUM_LAYERS'] == 101:
            ''' Load pretrained model '''
            x = tm.resnet101(pretrained=True)
        elif cfg['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
        elif cfg['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.norm_layer = norm_layer

        stage1_cfg = cfg['STAGE1']
        stage2_cfg = cfg['STAGE2']
        stage3_cfg = cfg['STAGE3']

        duc1 = self._make_duc_stage(stage1_cfg, 2048, 1024)
        duc2 = self._make_duc_stage(stage2_cfg, 1024, 512)
        duc3 = self._make_duc_stage(stage3_cfg, 512, self.conv_dim)

        self.duc = nn.Sequential(duc1, duc2, duc3)

        duc1_dense = self._make_duc_stage(stage1_cfg,2048,1024)
        duc2_dense = self._make_duc_stage(stage2_cfg,1024,512)
        duc3_dense = self._make_duc_stage(stage3_cfg,512,self.conv_dim)

        self.duc_dense = nn.Sequential(duc1_dense,duc2_dense,duc3_dense)

        self.conv_out = nn.Conv2d(
            self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

        self.conv_out_dense = nn.Conv2d(
            self.conv_dim,(self._preset_cfg['NUM_JOINTS_DENSE']-self._preset_cfg['NUM_JOINTS']),kernel_size=3,stride=1,padding=1)
        for params in self.preact.parameters():
            params.requires_grad = False
        for params in self.duc.parameters():
            params.requires_grad = False

    def forward(self, x):
        bk_out = self.preact(x)
        out = self.duc(bk_out)
        out_dense = self.duc_dense(bk_out)
        out = self.conv_out(out)
        out_dense = self.conv_out_dense(out_dense)
        out = torch.cat((out,out_dense),1)
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
        for m in self.duc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        #init dense-branch
        for m in self.duc_dense.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.conv_out_dense.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
