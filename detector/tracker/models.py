import os
from collections import defaultdict,OrderedDict

import torch.nn as nn

from tracker.utils.parse_config import *
from tracker.utils.utils import *
# from utils.syncbn import SyncBN
import time
import math

batch_norm=nn.BatchNorm2d

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, batch_norm(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])  # number of classes
            img_size = (int(hyperparams['width']),int(hyperparams['height']))
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, hyperparams['nID'], img_size, yolo_layer_count, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, nID, img_size, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.nID = nID # number of identities
        self.img_size = 0
        self.emb_dim = 512

        self.SmoothL1Loss  = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15*torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85*torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3*torch.ones(1))  # -2.3
        self.emb_scale = math.sqrt(2) * math.log(self.nID-1)
        

    def forward(self, p_cat,  img_size, targets=None, classifier=None, test_emb=False):
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nGh, nGw)

            self.grid_xy = self.grid_xy.to(p)
            self.anchor_wh = self.anchor_wh.to(p)

        p = p.view(nB, self.nA, self.nC + 5, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        
        p_emb = p_emb.permute(0,2,3,1).contiguous()
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

        # Training
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            else:
                tconf, tbox, tids = build_targets_thres(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            tconf, tbox, tids = tconf.cuda(), tbox.cuda(), tids.cuda()
            mask = tconf > 0

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.cuda.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf =  FT([0]), FT([0])
            lconf =  self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze().cuda()
            emb_mask,_ = mask.max(1)
            
            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            tids,_ = tids.max(1) 
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()
            
            if  test_emb:
                if np.prod(embedding.shape)==0  or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self. emb_dim+1).cuda()
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt
            
            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid =  self.IDLoss(logits, tids.squeeze())

            # Sum loss components
            loss = torch.exp(-self.s_r)*lbox + torch.exp(-self.s_c)*lconf + torch.exp(-self.s_id)*lid + \
                   (self.s_r + self.s_c + self.s_id)
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:,1,...].unsqueeze(-1)
            p_emb = p_emb.unsqueeze(1).repeat(1,self.nA,1,1,1).contiguous()
            p_cls = torch.zeros(nB,self.nA,nGh,nGw,1).to(p)              # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(nB, -1, p.shape[-1])


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=(1088, 608), nID=1591, test_emb=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['nID'] = nID
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.emb_dim = 512
        self.classifier = nn.Linear(self.emb_dim, nID)
        self.test_emb=test_emb


    def forward(self, x, targets=None, targets_len=None):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        #img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                if is_training:  # get loss
                    targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:  # get detections
                    x = module[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nT'] /= 3 
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values())).cuda()
        elif self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)


def create_grids(self, img_size, nGh, nGw):
    self.stride = img_size[0]/nGw
    assert self.stride == img_size[1] / nGh

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = torch.arange(nGh).repeat((nGw, 1)).transpose(0,1).view((1, 1, nGh, nGw)).float()
    #grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
