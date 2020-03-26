import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

import torch
import torchvision

from tools import random_heatmap

from alphapose.utils.config import update_config
from alphapose.utils.transforms import get_max_pred
from alphapose.utils.metrics import evaluate_mAP
from alphapose.models import builder


parser = argparse.ArgumentParser(description='my val')
parser.add_argument('--cfg', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--gpus', type=str)
parser.add_argument('--batch', type=int)
parser.add_argument('--flip-test', default=False, dest='flip_test', action='store_true')
parser.add_argument('--detector', dest='detector', default="yolo")

opt = parser.parse_args([
  '--cfg', './configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
  '--checkpoint', './models/fast_res50_256x192.pth',
  '--batch', '80',
  '--gpus', '0',
  '--flip-test'
])
cfg = update_config(opt.cfg)


gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")


# Demo heatmaps
# assume we have 14 kpts
size = (64, 48)

hms = np.stack([random_heatmap(size) for _ in range(14)], 0)



# validate_gt
# a Mscoco object
gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, cfg.DATA_PRESET, train=False)
eval_joints = gt_val_dataset.EVAL_JOINTS
gt_val_loader = torch.utils.data.DataLoader(
  gt_val_dataset, batch_size=opt.batch, shuffle=False, num_workers=20, drop_last=False
)


res = evaluate_mAP('./exp/json/validate_gt_kpt.json', ann_type='keypoints')

print(res['AP'])
