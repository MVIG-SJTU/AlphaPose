import argparse
import os
from os.path import split, join
import yaml
from easydict import EasyDict as edict

import torch
import torchvision

from alphapose.utils.config import update_config



parser = argparse.ArgumentParser(description='My demo')
parser.add_argument('--cfg', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--sp', default=False, action='store_true')
parser.add_argument('--detector', dest='detector', default="yolo")
parser.add_argument('--indir', dest='inputpath', default="")
parser.add_argument('--list', dest='inputlist', default="")
parser.add_argument('--image', dest='inputimg', default="")
parser.add_argument('--outdir', dest='outputpath', default="examples/res/")
parser.add_argument('--save_img', default=True, action='store_true')
parser.add_argument('--vis', default=False, action='store_true')
parser.add_argument('--profile', default=False, action='store_true')
parser.add_argument('--format', type=str)
parser.add_argument('--min_box_area', type=int, default=0)
parser.add_argument('--detbatch', type=int, default=5)
parser.add_argument('--posebatch', type=int, default=80)
parser.add_argument('--eval', dest='eval', default=False, action='store_true')
parser.add_argument('--gpus', type=str, dest='gpus', default="0")
parser.add_argument('--qsize', type=int, dest='qsize', default=1024)
parser.add_argument('--flip', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video', default="")
parser.add_argument('--webcam', dest='webcam', type=int, default=-1)
parser.add_argument('--save_video', dest='save_video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast')
parser.add_argument('--pose_track', dest='pose_track', action='store_true', default=False)


args = parser.parse_args([
  '--cfg', './configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
  '--checkpoint', './models/fast_res50_256x192.pth',
  '--indir', ''
])

print(args.cfg)

