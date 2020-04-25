# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
import argparse
import logging
import os
from types import MethodType

import torch

from .utils.config import update_config

parser = argparse.ArgumentParser(description='AlphaPose Training')

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--exp-id', default='default', type=str,
                    help='Experiment ID')

"----------------------------- General options -----------------------------"
parser.add_argument('--nThreads', default=60, type=int,
                    help='Number of data loading threads')
parser.add_argument('--snapshot', default=2, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.1.214:23345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                    help='job launcher')

"----------------------------- Training options -----------------------------"
parser.add_argument('--sync', default=False, dest='sync',
                    help='Use Sync Batchnorm', action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")

"----------------------------- Log options -----------------------------"
parser.add_argument('--board', default=True, dest='board',
                    help='Logging with tensorboard', action='store_true')
parser.add_argument('--debug', default=False, dest='debug',
                    help='Visualization debug', action='store_true')
parser.add_argument('--map', default=True, dest='map',
                    help='Evaluate mAP per epoch', action='store_true')


opt = parser.parse_args()
cfg_file_name = os.path.basename(opt.cfg)
cfg = update_config(opt.cfg)

cfg['FILE_NAME'] = cfg_file_name
cfg.TRAIN.DPG_STEP = [i - cfg.TRAIN.DPG_MILESTONE for i in cfg.TRAIN.DPG_STEP]
opt.world_size = cfg.TRAIN.WORLD_SIZE
opt.work_dir = './exp/{}-{}/'.format(opt.exp_id, cfg_file_name)
opt.gpus = [i for i in range(torch.cuda.device_count())]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")

if not os.path.exists("./exp/{}-{}".format(opt.exp_id, cfg_file_name)):
    os.makedirs("./exp/{}-{}".format(opt.exp_id, cfg_file_name))

filehandler = logging.FileHandler(
    './exp/{}-{}/training.log'.format(opt.exp_id, cfg_file_name))
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


def epochInfo(self, set, idx, loss, acc):
    self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
        set=set,
        idx=idx,
        loss=loss,
        acc=acc
    ))


logger.epochInfo = MethodType(epochInfo, logger)
