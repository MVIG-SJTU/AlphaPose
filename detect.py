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



if __name__ == '__main__':
  args, cfgs = get_args()

  detector = YOLODetector(cfg, args)

  images = glob('data/seedland/png_img/*')

  # used for debug
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


  # runtime = {
  #   'dt': []
  # }

  # # warmup
  # for i, image in enumerate(images):
  #   _ = detector.detect_one_img(image)
  #   if i >= 10:
  #     break


  # # calculate runtime
  # for i, image in enumerate(images):
  #   # img = cv2.imread(image)
  #   start_time = getTime()

  #   results = detector.detect_one_img(image)

  #   ckpt_time, det_time = getTime(start_time)
  #   runtime['dt'].append(det_time)

  #   # if results is None: continue

  #   # for res in results:
  #   #   bbox = [int(x) for x in res['bbox']]
  #   #   cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

  #   # cv2.imshow('_', img)
  #   # if cv2.waitKey(0) == 27: break

  # print('det time: {0:.4f}'.format(np.mean(runtime['dt'])))









