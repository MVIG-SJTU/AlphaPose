# encoding: utf-8
"""
@author:  tanghy
@contact: thutanghy@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
from ReidModels.ResNet import build_resnet_backbone
from ReidModels.bn_linear import BNneckLinear
class SpatialAttn(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,192
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),1,h,w)
        return z
class ResModel(nn.Module):

    def __init__(self, n_ID):
        super().__init__()
        self.backbone = build_resnet_backbone()
        self.head = BNneckLinear(n_ID)
        self.atten = SpatialAttn()
        self.conv1 = nn.Conv2d(17, 17, 1,stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.pool = nn.AvgPool2d(2, stride=2, padding=0,)
    def forward(self, input,posemap,map_weight):
        """
        See :class:`ReIDHeads.forward`.
        """
        feat = self.backbone(input)
        b,c,h,w = feat.shape
        att = self.conv1(torch.mul(posemap,map_weight))
        #print('att-1-size={}'.format(att.shape))
        att = F.relu(att)
        att = self.pool(att)
        att = self.conv1(att)
        #print('att-2-size={}'.format(att.shape))
        att = F.softmax(att)
        #print('att-3-size={}'.format(att.shape))
        att = self.atten(att)
        #print('att-4-size={}'.format(att.shape))
        att = att.expand(b,c,h,w)
        _feat = torch.mul(feat,att)
        feat = _feat + feat
        return self.head(feat)