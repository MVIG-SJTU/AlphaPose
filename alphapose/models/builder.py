from torch import nn

from alphapose.utils import Registry, build_from_cfg, retrieve_from_cfg


SPPE = Registry('sppe')
LOSS = Registry('loss')
DATASET = Registry('dataset')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_sppe(cfg, preset_cfg, **kwargs):
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, SPPE, default_args=default_args)


def build_loss(cfg):
    return build(cfg, LOSS)


def build_dataset(cfg, preset_cfg, **kwargs):
    exec(f'from ..datasets import {cfg.TYPE}')
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, DATASET, default_args=default_args)


def retrieve_dataset(cfg):
    exec(f'from ..datasets import {cfg.TYPE}')
    return retrieve_from_cfg(cfg, DATASET)
