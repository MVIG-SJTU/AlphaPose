# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from collections import namedtuple, OrderedDict

from nets.network import Network
from model.config import cfg

# The following is adapted from:
# https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.py

# Conv and DepthSepConv named tuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    DepthSepConv(kernel=3, stride=1, depth=64),
    DepthSepConv(kernel=3, stride=2, depth=128),
    DepthSepConv(kernel=3, stride=1, depth=128),
    DepthSepConv(kernel=3, stride=2, depth=256),
    DepthSepConv(kernel=3, stride=1, depth=256),
    DepthSepConv(kernel=3, stride=2, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    # use stride 1 for the 13th layer
    DepthSepConv(kernel=3, stride=1, depth=1024),
    DepthSepConv(kernel=3, stride=1, depth=1024)
]

def mobilenet_v1_base(final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None):
    """Mobilenet v1.

    Constructs a Mobilenet v1 network from inputs to the given final endpoint.

    Args:
        inputs: a tensor of shape [batch_size, height, width, channels].
        final_endpoint: specifies the endpoint to construct the network up to. It
            can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
            'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5_pointwise',
            'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
            'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
            'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        conv_defs: A list of ConvDef namedtuples specifying the net architecture.
        output_stride: An integer that specifies the requested ratio of input to
            output spatial resolution. If not None, then we invoke atrous convolution
            if necessary to prevent the network from reducing the spatial resolution
            of the activation maps. Allowed values are 8 (accurate fully convolutional
            mode), 16 (fast fully convolutional mode), 32 (classification mode).
        scope: Optional variable_scope.

    Returns:
        tensor_out: output tensor corresponding to the final_endpoint.
        end_points: a set of activations for external use, for example summaries or
                                losses.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
                                or depth_multiplier <= 0, or the target output_stride is not
                                allowed.
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = OrderedDict()

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    def conv_bn(in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def conv_dw(in_channels, kernel_size=3, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, (kernel_size - 1) // 2,\
                      groups=in_channels, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )

    def conv_pw(in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    in_channels = 3
    for i, conv_def in enumerate(conv_defs):
        end_point_base = 'Conv2d_%d' % i

        if output_stride is not None and current_stride == output_stride:
            # If we have reached the target output_stride, then we need to employ
            # atrous convolution with stride=1 and multiply the atrous rate by the
            # current unit's stride for use in subsequent layers.
            layer_stride = 1
            layer_rate = rate
            rate *= conv_def.stride
        else:
            layer_stride = conv_def.stride
            layer_rate = 1
            current_stride *= conv_def.stride

        out_channels = depth(conv_def.depth)
        if isinstance(conv_def, Conv):
            end_point = end_point_base
            end_points[end_point] = conv_bn(in_channels, out_channels, conv_def.kernel,
                                            stride=conv_def.stride)
            if end_point == final_endpoint:
                return nn.Sequential(end_points)

        elif isinstance(conv_def, DepthSepConv):
            end_points[end_point_base] = nn.Sequential(OrderedDict([
                ('depthwise', conv_dw(in_channels, conv_def.kernel, stride=layer_stride, dilation=layer_rate)),
                ('pointwise', conv_pw(in_channels, out_channels, 1, stride=1))]))

            if end_point_base + '_pointwise' == final_endpoint:
                return nn.Sequential(end_points)
        else:
            raise ValueError('Unknown convolution type %s for layer %d'
                                                % (conv_def.ltype, i))
        in_channels = out_channels
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

class mobilenetv1(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
    self._net_conv_channels = 512
    self._fc7_channels = 1024

  def init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      if m.__class__.__name__.find('Conv') == -1:
        return
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
      if m.bias is not None: m.bias.data.zero_()
      
    self.mobilenet.apply(lambda m: normal_init(m, 0, 0.09, True))
    normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_cls_score_net, 0, 0.01,  cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_bbox_pred_net, 0, 0.01,  cfg.TRAIN.TRUNCATED)
    normal_init(self.cls_score_net, 0, 0.01,  cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001,  cfg.TRAIN.TRUNCATED)

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv'] = net_conv

    return net_conv

  def _head_to_tail(self, pool5):
    fc7 = self._layers['tail'](pool5)
    fc7 = fc7.mean(3).mean(2)
    return fc7

  def _init_head_tail(self):
    self.mobilenet = mobilenet_v1_base()

    # Fix blocks  
    assert (0 <= cfg.MOBILENET.FIXED_LAYERS <= 12)
    for m in list(self.mobilenet.children())[:cfg.MOBILENET.FIXED_LAYERS]:
      for p in m.parameters():
        p.requires_grad = False
    
    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.mobilenet.apply(set_bn_fix)

    # Add weight decay
    def l2_regularizer(m, wd):
      if m.__class__.__name__.find('Conv') != -1:
        m.weight.weight_decay = cfg.MOBILENET.WEIGHT_DECAY
    if cfg.MOBILENET.REGU_DEPTH:
      self.mobilenet.apply(lambda x: l2_regularizer(x, cfg.MOBILENET.WEIGHT_DECAY))
    else:
      self.mobilenet.apply(lambda x: l2_regularizer(x, 0))
      # always set the first conv layer
      list(self.mobilenet.children())[0].apply(lambda x: l2_regularizer(x, cfg.MOBILENET.WEIGHT_DECAY))

    # Build mobilenet.
    self._layers['head'] = nn.Sequential(*list(self.mobilenet.children())[:12])
    self._layers['tail'] = nn.Sequential(*list(self.mobilenet.children())[12:])

  def load_pretrained_cnn(self, state_dict):
    # TODO
    print('Warning: No available pretrained model yet')
    return
    self.mobilenet.load_state_dict({k: state_dict[k] for k in list(self.resnet.state_dict())})
