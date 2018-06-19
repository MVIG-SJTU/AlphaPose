import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn import FPN50
from torch.autograd import Variable


class FPNSSD512(nn.Module):
    steps = (8, 16, 32, 64, 128, 256, 512)
    box_sizes = (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)  # default bounding box sizes for each feature map.
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))
    fm_sizes = (64, 32, 16, 8, 4, 2, 1)

    def __init__(self, num_classes):
        super(FPNSSD512, self).__init__()
        self.num_classes = num_classes
        self.extractor = FPN50()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()

        in_channels = 256
        num_anchors = (4, 6, 6, 6, 6, 4, 4)
        for i in range(len(num_anchors)):
        	self.loc_layers += [nn.Conv2d(in_channels, num_anchors[i]*4, kernel_size=3, padding=1)]
        	self.cls_layers += [nn.Conv2d(in_channels, num_anchors[i]*num_classes, kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0),-1,4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0,2,3,1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0),-1,self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


def test():
    net = SSD512(21)
    loc_preds, cls_preds = net(Variable(torch.randn(1,3,512,512)))
    print(loc_preds.size(), cls_preds.size())

# test()
