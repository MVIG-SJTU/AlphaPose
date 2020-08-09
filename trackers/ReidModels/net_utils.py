import os
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import pickle

from utils.log import logger


class ConcatAddTable(nn.Module):
    def __init__(self, *args):
        super(ConcatAddTable, self).__init__()
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        x_out = None
        for module in self._modules.values():
            x = module(input)
            if x_out is None:
                x_out = x
            else:
                x_out = x_out + x
        return x_out


def set_optimizer_state_devices(state, device_id=None):
    """
    set state in optimizer to a device. move to cpu if device_id==None
    :param state: optimizer.state
    :param device_id: None or a number
    :return:
    """
    for k, v in state.items():
        for k2 in v.keys():
            if hasattr(v[k2], 'cuda'):
                if device_id is None:
                    v[k2] = v[k2].cpu()
                else:
                    v[k2] = v[k2].cuda(device_id)

    return state


def save_net(fname, net, epoch=-1, optimizers=None, rm_prev_opt=False, max_n_ckpts=-1):
    import h5py
    with h5py.File(fname, mode='w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
        h5f.attrs['epoch'] = epoch

    if optimizers is not None:
        state_dicts = []
        for optimizer in optimizers:
            state_dict = deepcopy(optimizer.state_dict())
            state_dict['state'] = set_optimizer_state_devices(state_dict['state'], device_id=None)
            state_dicts.append(state_dict)

        state_file = fname + '.optimizer_state.pk'
        with open(state_file, 'wb') as f:
            pickle.dump(state_dicts, f)

        # remove
        if rm_prev_opt:
            root = os.path.split(fname)[0]
            for filename in os.listdir(root):
                filename = os.path.join(root, filename)
                if filename.endswith('.optimizer_state.pk') and filename != state_file:
                    logger.info(('Remove {}'.format(filename)))
                    os.remove(filename)

        # remove ckpt
        if max_n_ckpts > 0:
            root = os.path.split(fname)[0]
            ckpts = [fname for fname in os.listdir(root) if os.path.splitext(fname)[-1] == '.h5']
            ckpts = sorted(ckpts, key=lambda name: int(os.path.splitext(name)[0].split('_')[-1]))
            if len(ckpts) > max_n_ckpts:
                for ckpt in ckpts[0:-max_n_ckpts]:
                    filename = os.path.join(root, ckpt)
                    logger.info('Remove {}'.format(filename))
                    os.remove(filename)


def load_net(fname, net, prefix='', load_state_dict=False):
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        h5f_is_module = True
        for k in h5f.keys():
            if not str(k).startswith('module.'):
                h5f_is_module = False
                break
        if prefix == '' and not isinstance(net, nn.DataParallel) and h5f_is_module:
            prefix = 'module.'

        for k, v in net.state_dict().items():
            k = prefix + k
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                if v.size() != param.size():
                    logger.warning('Inconsistent shape: {}, {}'.format(v.size(), param.size()))
                else:
                    v.copy_(param)
            else:
                logger.warning('No layer: {}'.format(k))

        epoch = h5f.attrs['epoch'] if 'epoch' in h5f.attrs else -1

        if not load_state_dict:
            if 'learning_rates' in h5f.attrs:
                lr = h5f.attrs['learning_rates']
            else:
                lr = h5f.attrs.get('lr', -1)
                lr = np.asarray([lr] if lr > 0 else [], dtype=np.float)

            return epoch, lr

        state_file = fname + '.optimizer_state.pk'
        if os.path.isfile(state_file):
            with open(state_file, 'rb') as f:
                state_dicts = pickle.load(f)
                if not isinstance(state_dicts, list):
                    state_dicts = [state_dicts]
        else:
            state_dicts = None
        return epoch, state_dicts


def is_cuda(model):
    p = next(model.parameters())
    return p.is_cuda


def get_device(model):
    if is_cuda(model):
        p = next(model.parameters())
        return p.get_device()
    else:
        return None


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
