from __future__ import absolute_import, division, print_function

import os

import torch
import torch.nn as nn
import torchvision.models as models

from .backbones.darknet import darknet53
from .backbones.dlav0 import get_pose_net as get_dlav0
from .backbones.hardnet import get_hard_net
from .backbones.large_hourglass import get_large_hourglass_net
from .backbones.mobilenet.mobilenetv3 import get_mobile_pose_netv3
from .backbones.mobilenet.mobilenetv2 import get_mobile_pose_netv2
from .backbones.msra_resnet import get_resnet
from .backbones.pose_dla_dcn import get_pose_net as get_dla_dcn
from .backbones.pose_higher_hrnet import get_hrpose_net
from .backbones.resnet_dcn import get_pose_net as get_pose_net_dcn
from .backbones.shufflenetv2_dcn import get_shufflev2_net
from .backbones.ghost_net import get_ghost_net
from .backbones.efficientdet import get_efficientdet
from .heads.keypoint import KeypointHead

_backbone_factory = {
  'res': get_resnet, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'resdcn': get_pose_net_dcn,
  'hourglass': get_large_hourglass_net,
  'mobilenetv3': get_mobile_pose_netv3,
  'mobilenetv2': get_mobile_pose_netv2,  
  'shufflenetV2': get_shufflev2_net,
  'hrnet': get_hrpose_net,
  'hardnet': get_hard_net,
  'darknet': darknet53,
  'ghostnet': get_ghost_net,
  'efficientdet':get_efficientdet,
}

_head_factory = {
  'keypoint': KeypointHead
}

class BackBoneWithHead(nn.Module):

    def __init__(self, arch, head_conv, cfg):
        super(BackBoneWithHead, self).__init__()    
        
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        arch = arch[:arch.find('_')] if '_' in arch else arch
        backbone = _backbone_factory[arch]
        self.backbone_model = backbone(num_layers=num_layers, cfg = cfg)
        
        head = _head_factory[cfg.MODEL.HEADS_NAME]
        self.head_model = head(cfg.MODEL.INTERMEDIATE_CHANNEL, cfg.MODEL.HEAD_CONV)

    def forward(self, x):
        x = self.backbone_model(x)
        return self.head_model(x)



def create_model(arch, head_conv, cfg):
   
    return BackBoneWithHead(arch, head_conv, cfg)

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
  
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '\
                      'loaded shape{}. {}'.format(
                  k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
          'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
