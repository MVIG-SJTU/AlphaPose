# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
import numpy.random as npr

import torch

def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
  """A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report
  """
  rpn_top_n = cfg.TEST.RPN_TOP_N

  scores = rpn_cls_prob[:, :, :, num_anchors:]

  rpn_bbox_pred = rpn_bbox_pred.view(-1, 4)
  scores = scores.contiguous().view(-1, 1)

  length = scores.size(0)
  if length < rpn_top_n:
    # Random selection, maybe unnecessary and loses good proposals
    # But such case rarely happens
    top_inds = torch.from_numpy(npr.choice(length, size=rpn_top_n, replace=True)).long().cuda()
  else:
    top_inds = scores.sort(0, descending=True)[1]
    top_inds = top_inds[:rpn_top_n]
    top_inds = top_inds.view(rpn_top_n)

  # Do the selection here
  anchors = anchors[top_inds, :].contiguous()
  rpn_bbox_pred = rpn_bbox_pred[top_inds, :].contiguous()
  scores = scores[top_inds].contiguous()

  # Convert anchors into proposals via bbox transformations
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

  # Clip predicted boxes to image
  proposals = clip_boxes(proposals, im_info[:2])

  # Output rois blob
  # Our RPN implementation only supports a single input image, so all
  # batch inds are 0
  batch_inds = proposals.data.new(proposals.size(0), 1).zero_()
  blob = torch.cat([batch_inds, proposals], 1)
  return blob, scores
