import torch
import os
from collections import OrderedDict
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module'):
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        print("=> Loaded checkpoint '{}'".format(checkpoint_path))
    else:
        print("=> Error: No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(model, url, filter_fn=None, strict=True):
    if not url:
        print("=> Warning: Pretrained model URL is empty, using random initialization.")
        return
    state_dict = load_state_dict_from_url(url, progress=False, map_location='cpu')
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)
    model.load_state_dict(state_dict, strict=strict)
