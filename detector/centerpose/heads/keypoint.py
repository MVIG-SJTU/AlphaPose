from __future__ import absolute_import, division, print_function

import os

import torch
import torch.nn as nn

        
class KeypointHead(nn.Module):

    def __init__(self, intermediate_channel, head_conv):
        super(KeypointHead, self).__init__()    
        
        self.hm = nn.Sequential(
                    nn.Conv2d(intermediate_channel, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0))
        self.wh = nn.Sequential(
                    nn.Conv2d(intermediate_channel, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0))
        self.hps = nn.Sequential(
                    nn.Conv2d(intermediate_channel, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 34, kernel_size=1, stride=1, padding=0))                  
        self.reg = nn.Sequential(
                    nn.Conv2d(intermediate_channel, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0))        
        self.hm_hp = nn.Sequential(
                    nn.Conv2d(intermediate_channel, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 17, kernel_size=1, stride=1, padding=0))                                           
        self.hp_offset = nn.Sequential(
                    nn.Conv2d(intermediate_channel, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0))                      
        self.init_weights()
                                           
    def forward(self, x):
        
        return [self.hm(x), self.wh(x), self.hps(x), self.reg(x), self.hm_hp(x), self.hp_offset(x)]
                        
    def init_weights(self):
        self.hm[-1].bias.data.fill_(-2.19)     
        self.hm_hp[-1].bias.data.fill_(-2.19)                                    
        self.fill_fc_weights(self.wh)
        self.fill_fc_weights(self.hps)
        self.fill_fc_weights(self.reg)
        self.fill_fc_weights(self.hp_offset)        
        
    def fill_fc_weights(self, layers):
      for m in layers.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.normal_(m.weight, std=0.001)
          if m.bias is not None:
            nn.init.constant_(m.bias, 0)                              
