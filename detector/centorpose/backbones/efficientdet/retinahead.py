import numpy as np
import torch.nn as nn

from .conv_module import ConvModule, bias_init_with_prob, normal_init
from six.moves import map, zip

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

class RetinaHead(nn.Module):
    """
    An anchor-based head used in [1]_.
    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.
    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf
    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                num_classes,
                 in_channels,
                 feat_channels=64,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(RetinaHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        self.cls_out_channels = num_classes
        self._init_layers()
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        #self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        #self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        #normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        #reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        #for reg_conv in self.reg_convs:
        #    reg_feat = reg_conv(reg_feat)
        
        cls_score = self.retina_cls(cls_feat)
        # out is B x C x W x H, with C = n_classes + n_anchors
        #cls_score = cls_score.permute(0, 2, 3, 1)
        #batch_size, width, height, channels = cls_score.shape
        #cls_score = cls_score.view(batch_size, width, height, self.num_anchors, self.num_classes)
        #cls_score = cls_score.contiguous().view(x.size(0), -1, self.num_classes)


        #bbox_pred = self.retina_reg(reg_feat)
        #bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        #bbox_pred = bbox_pred.contiguous().view(bbox_pred.size(0), -1, 4)
        return [cls_score]
    def forward(self, feats):
        return multi_apply(self.forward_single, feats)
