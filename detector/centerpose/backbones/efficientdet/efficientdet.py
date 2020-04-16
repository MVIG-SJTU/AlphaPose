import torch
import torch.nn as nn
import math
from .efficientnet import EfficientNet
from .bifpn import BIFPN
from .retinahead import RetinaHead
from torchvision.ops import nms 
import torch.nn.functional as F

MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}
class EfficientDet(nn.Module):
    def __init__(self,
                 intermediate_channels,
                 network = 'efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=32,
                 D_class=3,
                 scale_ratios = [0.5, 1, 2, 4, 8,16,32],
                 ):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        self.neck = BIFPN(in_channels=self.backbone.get_list_features(),
                                out_channels=W_bifpn,
                                stack=D_bifpn,
                                num_outs=7)
        self.bbox_head = RetinaHead(num_classes = intermediate_channels,
                                    in_channels = W_bifpn)

        self.scale_ratios = scale_ratios
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()

    def forward(self, inputs):
        x = self.extract_feat(inputs)
        outs = self.bbox_head(x)

        return outs[0][1]
        
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x
    
