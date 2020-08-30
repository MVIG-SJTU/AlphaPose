from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.SE_Resnet import SEResnet
from .layers.ShuffleResnet import ShuffleResnet

BN_MOMENTUM = 0.1
DEBUG = False

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels, norm_layer, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  norm_layer, stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        if DEBUG:
          print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                          stride=stride, padding=1, groups=groups, bias=bias))

        self.add_module('norm', norm_layer(groups, momentum=BN_MOMENTUM))
    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer, kernel=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.out_channels = out_channels
        out_ch = out_channels
        groups = 1
        if DEBUG:
          print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        pad = kernel//2 if padding == 0 else padding
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=pad, groups=groups, bias=bias))
        self.add_module('norm', norm_layer(out_ch, momentum=BN_MOMENTUM))
        self.add_module('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)


class BRLayer(nn.Sequential):
    def __init__(self, in_channels, norm_layer):
        super().__init__()

        self.add_module('norm', norm_layer(in_channels))
        self.add_module('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, norm_layer, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.norm_layer = norm_layer
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0

        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch, norm_layer))
          else:
            layers_.append(ConvLayer(inch, outch, norm_layer))

          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        if DEBUG:
          print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class HarDBlock_v2(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.insert(0, k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, norm_layer, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.norm_layer = norm_layer
        self.out_partition = collections.defaultdict(list)

        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          for j in link:
            self.out_partition[j].append(outch)

        cur_ch = in_channels
        for i in range(n_layers):
          accum_out_ch = sum( self.out_partition[i] )
          real_out_ch = self.out_partition[i][0]
          conv_layers_.append( nn.Conv2d(cur_ch, accum_out_ch, kernel_size=3, stride=1, padding=1, bias=True) )
          bnrelu_layers_.append( BRLayer(real_out_ch, norm_layer) )
          cur_ch = real_out_ch
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += real_out_ch
        self.conv_layers = nn.ModuleList(conv_layers_)
        self.bnrelu_layers = nn.ModuleList(bnrelu_layers_)

    def transform(self, blk, trt=False):
        # Transform weight matrix from a pretrained HarDBlock v1
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [blk.layers[k-1][0].weight.shape[0] if k > 0 else
                       blk.layers[0  ][0].weight.shape[1] for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias


            self.conv_layers[i].weight[0:part[0], :, :,:] = w_src[:, 0:in_ch, :,:]
            self.layer_bias.append(b_src)
            #if b_src is not None:
            #  self.layer_bias[i] = b_src.view(1,-1,1,1)
            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    #for pytorch, add bias with standalone tensor is more efficient than within conv.bias
                    #this is because the amount of non-zero bias is small, 
                    #but if we use conv.bias, the number of bias will be much larger
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None 


            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link) ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chos = sum( self.out_partition[ly][0:part_id] )
                    choe = chos + part[0]
                    chis = sum( link_ch[0:j] )
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :,:,:] = w_src[:, chis:chie,:,:]

            #update BatchNorm or remove it if there is no BatchNorm in the v1 block
            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], self.norm_layer):
                self.bnrelu_layers[i] = nn.Sequential(
                         blk.layers[i][1],
                         blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]

            xout = self.conv_layers[i](xin)
            layers_.append(xout)

            xin = xout[:,0:part[0],:,:] if len(part) > 1 else xout
            if self.layer_bias[i] is not None:
                xin += self.layer_bias[i].view(1,-1,1,1)

            if len(link) > 1:
                for j in range( len(link) - 1 ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chs = sum( self.out_partition[ly][0:part_id] )
                    che = chs + part[0]

                    xin += layers_[ly][:,chs:che,:,:]

            xin = self.bnrelu_layers[i](xin)

            if i%2 == 0 or i == len(self.conv_layers)-1:
              outs_.append(xin)

        out = torch.cat(outs_, 1)
        return out


class HarDNetBase(nn.Module):
    def __init__(self, arch, norm_layer, depth_wise=False):
        super().__init__()
        if arch == 85:
          first_ch  = [48, 96]
          second_kernel = 3

          ch_list = [  192, 256, 320, 480, 720]
          grmul = 1.7
          gr       = [  24, 24, 28, 36, 48]
          n_layers = [   8, 16, 16, 16, 16]
        elif arch == 68:
          first_ch  = [32, 64]
          second_kernel = 3

          ch_list = [  128, 256, 320, 640]
          grmul = 1.7
          gr       = [  14, 16, 20, 40]
          n_layers = [   8, 16, 16, 16]
        else:
          print("Error: HarDNet",arch," has no implementation.")
          exit()

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], norm_layer=norm_layer, kernel=3,
                       stride=2,  bias=False) )

        # Second Layer
        self.base.append ( ConvLayer(first_ch[0], first_ch[1], norm_layer, kernel=second_kernel) )

        # Maxpooling or DWConv3x3 downsampling
        self.base.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], norm_layer, dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append ( blk )

            if i != blks-1:            
              self.base.append ( ConvLayer(ch, ch_list[i], norm_layer, kernel=1) )
            ch = ch_list[i]
            if i== 0:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
            elif i != blks-1 and i != 1 and i != 3:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))
        

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_uniform_(m.state_dict()[key], nonlinearity='relu')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True)
        if concat:
          out = torch.cat([out, skip], 1)
        return out

@SPPE.register_module
class HarDNetPose(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HarDNetPose, self).__init__()
        assert cfg['DOWN_RATIO'] in [2, 4, 8, 16]
        self.norm_layer = norm_layer
        self._preset_cfg = cfg['PRESET']
        self.first_level = int(np.log2(cfg['DOWN_RATIO']))-1
        self.trt = cfg['TRT']
        
        self.base = HarDNetBase(cfg['NUM_LAYERS'], norm_layer).base
        self.last_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        if cfg['NUM_LAYERS'] == 85:
          self.last_proj = ConvLayer(784, 256, norm_layer, kernel=1)
          self.last_blk = HarDBlock(768, 80, 1.7, 8, norm_layer)
          self.skip_nodes = [1,3,8,13]
          self.SC = [32, 32, 0]
          gr = [64, 48, 28]
          layers = [8, 8, 4]
          ch_list2 = [224 + self.SC[0], 160 + self.SC[1], 96 + self.SC[2]]
          channels = [96, 214, 458, 784]
          self.skip_lv = 3
          scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        elif cfg['NUM_LAYERS'] == 68:
          self.last_proj = ConvLayer(654, 192, norm_layer, kernel=1)
          self.last_blk = HarDBlock(576, 72, 1.7, 8, norm_layer)
          self.skip_nodes = [1,3,8,11]
          self.SC = [32, 32, 0 ]  
          gr = [48, 32, 20]
          layers = [8, 8, 4]
          ch_list2 = [224+self.SC[0], 96+self.SC[1], 64+self.SC[2]]
          channels = [64, 124, 328, 654]
          self.skip_lv = 2
          scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        self.avg9x9   = nn.AvgPool2d(kernel_size=(9,9), stride=1, padding=(4,4))
        prev_ch = self.last_blk.get_out_ch()

        for i in range(3):
            skip_ch = channels[3-i]
            self.transUpBlocks.append(TransitionUp(prev_ch, prev_ch))
            if i < self.skip_lv:
              cur_ch = prev_ch + skip_ch
            else:
              cur_ch = prev_ch
            self.conv1x1_up.append(ConvLayer(cur_ch, ch_list2[i], norm_layer, kernel=1))
            cur_ch = ch_list2[i]
            cur_ch -= self.SC[i]
            cur_ch *= 3

            blk = HarDBlock(cur_ch, gr[i], 1.7, layers[i], norm_layer)

            self.denseBlocksUp.append(blk)
            prev_ch = blk.get_out_ch()
        
        prev_ch += self.SC[0] + self.SC[1] + self.SC[2]

        weights_init(self.denseBlocksUp) 
        weights_init(self.conv1x1_up) 
        weights_init(self.last_blk)
        weights_init(self.last_proj)
        
        out_channel = self._preset_cfg['NUM_JOINTS']

        ch = max(128, out_channel*4)
        self.conv_out = nn.Sequential(
            nn.Conv2d(prev_ch, ch,
              kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_channel, 
              kernel_size=cfg['FINAL_CONV_KERNEL'], stride=1, 
              padding=cfg['FINAL_CONV_KERNEL'] // 2, bias=True))
        fill_fc_weights(self.conv_out)
        self.conv_out[-1].bias.data.fill_(-2.19)


    def v2_transform(self):
        print('Transform HarDBlock v2..')
        for i in range( len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers, blk.norm_layer)
                self.base[i].transform(blk, self.trt)
        blk = self.last_blk
        self.last_blk = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers, blk.norm_layer)
        self.last_blk.transform(blk, self.trt)
        for i in range(3):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers, blk.norm_layer)
            self.denseBlocksUp[i].transform(blk, self.trt)

    def forward(self, x):
        xs = []
        x_sc = []
        
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.skip_nodes:
                xs.append(x)
        
        x = self.last_proj(x)
        x = self.last_pool(x)
        x2 = self.avg9x9(x)
        x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
        x = torch.cat([x,x2,x3],1)
        x = self.last_blk(x)
        
        for i in range(3):
            skip_x = xs[3-i]
            x = self.transUpBlocks[i](x, skip_x, (i<self.skip_lv))
            x = self.conv1x1_up[i](x)
            if self.SC[i] > 0:
              end = x.shape[1]
              x_sc.append( x[:,end-self.SC[i]:,:,:].contiguous() )
              x = x[:,:end-self.SC[i],:,:].contiguous()
            x2 = self.avg9x9(x)
            x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
            x = torch.cat([x,x2,x3],1)
            x = self.denseBlocksUp[i](x)

        scs = [x]
        for i in range(3):
          if self.SC[i] > 0:
            scs.insert(0,  F.interpolate(
                            x_sc[i], size=(x.size(2), x.size(3)), 
                            mode="bilinear", align_corners=True) )
        x = torch.cat(scs,1)
        x = self.conv_out(x)
        return x
    
    def _initialize(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, self.norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
  model = HarDNetPose(cfg, **kwargs)
  if is_train and cfg.MODEL.INIT_WEIGHTS:
        model._initialize(cfg.MODEL.INIT_WEIGHTS)
  total_params = sum(p.numel() for p in model.parameters())
  print( "Parameters=", total_params )
  return model
