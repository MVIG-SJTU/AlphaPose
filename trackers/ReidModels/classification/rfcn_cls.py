import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import net_utils
from models.backbone.sqeezenet import DilationLayer, FeatExtractorSqueezeNetx16
from models.psroi_pooling.modules.psroi_pool import PSRoIPool


class Model(nn.Module):
    feat_stride = 4

    def __init__(self, extractor='squeezenet', pretrained=False, transform_input=False):
        super(Model, self).__init__()

        if extractor == 'squeezenet':
            feature_extractor = FeatExtractorSqueezeNetx16(pretrained)
        else:
            assert False, 'invalid feature extractor: {}'.format(extractor)

        self.feature_extractor = feature_extractor

        in_channels = self.feature_extractor.n_feats[-1]
        self.stage_0 = nn.Sequential(
            nn.Dropout2d(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        n_feats = self.feature_extractor.n_feats[1:]
        in_channels = 256
        out_cs = [128, 256]
        for i in range(1, len(n_feats)):
            out_channels = out_cs[-i]
            setattr(self, 'upconv_{}'.format(i),
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ))

            feat_channels = n_feats[-1-i]
            setattr(self, 'proj_{}'.format(i), nn.Sequential(
                net_utils.ConcatAddTable(
                    DilationLayer(feat_channels, out_channels // 2, 3, dilation=1),
                    DilationLayer(feat_channels, out_channels // 2, 5, dilation=1),
                ),
                nn.Conv2d(out_channels // 2, out_channels // 2, 1),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels + out_channels // 2

        roi_size = 7
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, roi_size * roi_size, 1, padding=1)
        )
        self.psroipool_cls = PSRoIPool(roi_size, roi_size, 1./self.feat_stride, roi_size, 1)
        self.avg_pool = nn.AvgPool2d(roi_size, roi_size)

    def get_cls_score(self, cls_feat, rois):
        """

        :param cls_feat: [N, rsize*rsize, H, W]
        :param rois: [N, 5] (batch_id, x1, y1, x2, y2)
        :return: [N], [N]
        """
        # rois = rois / 8.
        cls_scores = self.psroipool_cls(cls_feat, rois)
        cls_scores = self.avg_pool(cls_scores).view(-1)
        cls_probs = torch.sigmoid(cls_scores)
        return cls_scores, cls_probs

    def get_cls_score_numpy(self, cls_feat, rois):
        """

        :param cls_feat: [1, rsize*rsize, H, W]
        :param rois: numpy array [N, 4] ( x1, y1, x2, y2)
        :return: [N], [N]
        """
        n_rois = rois.shape[0]
        if n_rois <= 0:
            return np.empty([0])

        _rois = np.zeros([n_rois, 5], dtype=np.float32)
        _rois[:, 1:5] = rois.astype(np.float32)
        _rois = Variable(torch.from_numpy(_rois)).cuda(cls_feat.get_device())

        cls_scores = self.psroipool_cls(cls_feat, _rois)
        cls_scores = self.avg_pool(cls_scores).view(-1)
        cls_probs = torch.sigmoid(cls_scores).data.cpu().numpy()

        return cls_probs

    def forward(self, x, gts=None):
        feats = self.feature_extractor(x)
        x_in = self.stage_0(feats[-1])

        # up conv
        n_feats = self.feature_extractor.n_feats[1:]
        for i in range(1, len(n_feats)):
            x_depth_out = getattr(self, 'upconv_{}'.format(i))(x_in)
            x_project = getattr(self, 'proj_{}'.format(i))(feats[-1-i])
            x_in = torch.cat((x_depth_out, x_project), 1)

        # cls features
        x_cls_in = x_in
        # x_cls_in = F.dropout2d(x_cls_in, training=self.training, inplace=True)
        cls_feat = self.cls_conv(x_cls_in)

        return cls_feat
