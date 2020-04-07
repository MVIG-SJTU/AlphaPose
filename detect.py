import os
from os.path import exists, split, join
from glob import glob
import sys
sys.path.insert(0, 'detector')

import numpy as np

import cv2

import torch


from detector.yolo.preprocess import prep_image
from detector.apis import BaseDetector
from detector.yolo_api import YOLODetector
from detector.yolo_cfg import cfg
from alphapose.utils.vis import getTime

from demo import get_args


def benchmark_on_single_image():
  # bench mark on 
  img_source = './data/seedland/pose_seg_hard/1.jpg'
  # results = detector.detect_one_img(img_source)
  img = detector.image_preprocess(img_source)
  if img.dim() == 3:
    img = img.unsqueeze(0)
  orig_img = cv2.imread(img_source)[:, :, ::-1]
  im_dim_list = orig_img.shape[1], orig_img.shape[0]
  
  imgs = torch.cat([img])
  im_dim_list = torch.FloatTensor([im_dim_list]).repeat(1, 2)
  orig_imgs = [orig_img]

  print(im_dim_list)
  dets = detector.images_detection(img, im_dim_list)
  dets = dets.cpu().numpy()
  print(dets.shape)

  idx = dets[:, 0].astype(np.int)
  boxes = dets[:, 1:5]
  scores = dets[:, 5:6]

  print(idx)
  print(boxes)
  print(scores)
  orig_img = np.ascontiguousarray(orig_img[:, :, ::-1])

  for box in boxes:
    cv2.rectangle(orig_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)

  cv2.imshow('_', orig_img)
  cv2.waitKey(0)



if __name__ == '__main__':
  args, cfgs = get_args()

  detector = YOLODetector(cfg, args)

  # used for debug, we pick pose_seg_hard/1.jpg 2.jpg as benchmark images
  img_sources = [
    './data/seedland/pose_seg_hard/1.jpg',
    './data/seedland/pose_seg_hard/2.jpg'
  ]

  imgs = torch.cat([detector.image_preprocess(_) for _ in img_sources])
  orig_imgs = [cv2.imread(_) for _ in img_sources]
  im_dim_list = torch.FloatTensor([(_.shape[1], _.shape[0]) for _ in orig_imgs]).repeat(1, 2)

  print(imgs.shape)
  print(im_dim_list)
  """ im_dim_list should be:
  tensor([
    [614., 728., 614., 728.],
    [550., 662., 550., 662.]
  ])
  """
  
  dets = detector.images_detection(imgs, im_dim_list).cpu().numpy()

  print(dets.shape)
  print(dets[:, 0])
  print(dets[:, 1:5])
  print(dets[:, 5:])

  
  """ The benchmark results should be (astype(int)):
  batch index, box, confidence, 
 [[  0 101 208 507 723   0   0   0]
  [  0 245   3 578 726   0   0   0]
  [  0 249   0 575 453   0   0   0]
  [  1 123  53 296 619   0   0   0]
  [  1  32  14 167 583   0   0   0]
  [  1 354  74 534 604   0   0   0]
  [  1 256  84 509 585   0   0   0]
  [  1 265  72 418 596   0   0   0]]
  """










  









