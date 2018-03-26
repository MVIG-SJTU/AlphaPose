import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from collections import OrderedDict
import re
import torch

import argparse
parser = argparse.ArgumentParser(description='Convert tf-faster-rcnn model to pytorch-faster-rcnn model')
parser.add_argument('--tensorflow_model',
                    help='the path of tensorflow_model',
                    default=None, type=str)

args = parser.parse_args()

reader = pywrap_tensorflow.NewCheckpointReader(args.tensorflow_model)
var_to_shape_map = reader.get_variable_to_shape_map()
var_dict = {k:reader.get_tensor(k) for k in var_to_shape_map.keys()}

del var_dict['Variable']

for k in list(var_dict.keys()):
    if 'Momentum' in k:
        del var_dict[k]

for k in list(var_dict.keys()):
    if k.find('/') >= 0:
        var_dict['mobilenet' + k[k.find('/'):]] = var_dict[k]
        del var_dict[k]

dummy_replace = OrderedDict([
                ('moving_mean', 'running_mean'),\
                ('moving_variance', 'running_var'),\
                ('weights', 'weight'),\
                ('biases', 'bias'),\
                ('/BatchNorm', '.1'),\
                ('_pointwise/', '.pointwise.0.'),\
                ('_depthwise/depthwise_', '.depthwise.0.'),\
                ('_pointwise.1', '.pointwise.1'),\
                ('_depthwise.1', '.depthwise.1'),\
                ('Conv2d_0/', 'Conv2d_0.0.'),\
                ('mobilenet/rpn_conv/3x3', 'rpn_net'),\
                ('mobilenet/rpn_cls_score', 'rpn_cls_score_net'),\
                ('mobilenet/cls_score', 'cls_score_net'),\
                ('mobilenet/rpn_bbox_pred', 'rpn_bbox_pred_net'),\
                ('mobilenet/bbox_pred', 'bbox_pred_net'),\
                ('gamma', 'weight'),\
                ('beta', 'bias'),\
                ('/', '.')])

for a, b in dummy_replace.items():
    for k in list(var_dict.keys()):
        if a in k:
            var_dict[k.replace(a,b)] = var_dict[k]
            del var_dict[k]

# print set(var_dict.keys()) - set(x.keys())
# print set(x.keys()) - set(var_dict.keys())

for k in list(var_dict.keys()):
    if var_dict[k].ndim == 4:
        if 'depthwise' in k:
            var_dict[k] = var_dict[k].transpose((2, 3, 0, 1)).copy(order='C')
        else:
            var_dict[k] = var_dict[k].transpose((3, 2, 0, 1)).copy(order='C')
    if var_dict[k].ndim == 2:
        var_dict[k] = var_dict[k].transpose((1, 0)).copy(order='C')
    # assert x[k].shape == var_dict[k].shape, k

for k in list(var_dict.keys()):
    var_dict[k] = torch.from_numpy(var_dict[k])


torch.save(var_dict, args.tensorflow_model[:args.tensorflow_model.find('.ckpt')]+'.pth')
