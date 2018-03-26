# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

def bbox_transform(ex_rois, gt_rois):
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = torch.log(gt_widths / ex_widths)
  targets_dh = torch.log(gt_heights / ex_heights)

  targets = torch.stack(
    (targets_dx, targets_dy, targets_dw, targets_dh), 1)
  return targets


def bbox_transform_inv(boxes, deltas):
  # Input should be both tensor or both Variable and on the same device
  if len(boxes) == 0:
    return deltas.detach() * 0

  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = deltas[:, 0::4]
  dy = deltas[:, 1::4]
  dw = deltas[:, 2::4]
  dh = deltas[:, 3::4]
  
  pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
  pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
  pred_w = torch.exp(dw) * widths.unsqueeze(1)
  pred_h = torch.exp(dh) * heights.unsqueeze(1)

  pred_boxes = torch.cat(\
    [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,\
                              pred_ctr_y - 0.5 * pred_h,\
                              pred_ctr_x + 0.5 * pred_w,\
                              pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)

  return pred_boxes


def clip_boxes(boxes, im_shape):
  """
  Clip boxes to image boundaries.
  boxes must be tensor or Variable, im_shape can be anything but Variable
  """

  if not hasattr(boxes, 'data'):
    boxes_ = boxes.numpy()

  boxes = boxes.view(boxes.size(0), -1, 4)
  boxes = torch.stack(\
    [boxes[:,:,0].clamp(0, im_shape[1] - 1),
     boxes[:,:,1].clamp(0, im_shape[0] - 1),
     boxes[:,:,2].clamp(0, im_shape[1] - 1),
     boxes[:,:,3].clamp(0, im_shape[0] - 1)], 2).view(boxes.size(0), -1)

  return boxes
