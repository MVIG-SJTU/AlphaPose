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
        var_dict['vgg' + k[k.find('/'):]] = var_dict[k]
        del var_dict[k]

dummy_replace = OrderedDict([
                ('weights', 'weight'),\
                ('biases', 'bias'),\
                ('vgg/rpn_conv/3x3', 'rpn_net'),\
                ('vgg/rpn_cls_score', 'rpn_cls_score_net'),\
                ('vgg/cls_score', 'cls_score_net'),\
                ('vgg/rpn_bbox_pred', 'rpn_bbox_pred_net'),\
                ('vgg/bbox_pred', 'bbox_pred_net'),\
                ('/', '.')])

for a, b in dummy_replace.items():
    for k in list(var_dict.keys()):
        if a in k:
            var_dict[k.replace(a,b)] = var_dict[k]
            del var_dict[k]

layer_map = OrderedDict([
    ('conv1.conv1_1', 'features.0'),\
    ('conv1.conv1_2', 'features.2'),\
    ('conv2.conv2_1', 'features.5'),\
    ('conv2.conv2_2', 'features.7'),\
    ('conv3.conv3_1', 'features.10'),\
    ('conv3.conv3_2', 'features.12'),\
    ('conv3.conv3_3', 'features.14'),\
    ('conv4.conv4_1', 'features.17'),\
    ('conv4.conv4_2', 'features.19'),\
    ('conv4.conv4_3', 'features.21'),\
    ('conv5.conv5_1', 'features.24'),\
    ('conv5.conv5_2', 'features.26'),\
    ('conv5.conv5_3', 'features.28'),\
    ('fc6', 'classifier.0'),\
    ('fc7', 'classifier.3')])

for a, b in layer_map.items():
    for k in list(var_dict.keys()):
        if a in k:
            var_dict[k.replace(a,b)] = var_dict[k]
            del var_dict[k]

for k in list(var_dict.keys()):
    if 'classifier.0' in k:
        if var_dict[k].ndim == 2: # weight
            var_dict[k] = var_dict[k].reshape(7,7,512,4096).transpose((3, 2, 0, 1)).reshape(4096, -1).copy(order='C')
    else:
        if var_dict[k].ndim == 4:
            var_dict[k] = var_dict[k].transpose((3, 2, 0, 1)).copy(order='C')
        if var_dict[k].ndim == 2:
            var_dict[k] = var_dict[k].transpose((1, 0)).copy(order='C')
    # assert x[k].shape == var_dict[k].shape, k

for k in list(var_dict.keys()):
    var_dict[k] = torch.from_numpy(var_dict[k])

torch.save(var_dict, args.tensorflow_model[:args.tensorflow_model.find('.ckpt')]+'.pth')
