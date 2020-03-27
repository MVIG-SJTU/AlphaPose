import os
from os.path import exists, split, join
from glob import glob

import numpy as np

import cv2

from detector.apis import BaseDetector
from detector.yolo_api import YOLODetector
from detector.yolo_cfg import cfg
from alphapose.utils.vis import getTime

from demo import get_args



if __name__ == '__main__':
  args, cfgs = get_args()

  detector = YOLODetector(cfg, args)

  images = glob('data/seedland/png_img/*')

  runtime = {
    'dt': []
  }

  # warmup
  for i, image in enumerate(images):
    _ = detector.detect_one_img(image)
    if i >= 10:
      break


  # calculate runtime
  for i, image in enumerate(images):
    # img = cv2.imread(image)
    start_time = getTime()

    results = detector.detect_one_img(image)

    ckpt_time, det_time = getTime(start_time)
    runtime['dt'].append(det_time)

    # if results is None: continue

    # for res in results:
    #   bbox = [int(x) for x in res['bbox']]
    #   cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

    # cv2.imshow('_', img)
    # if cv2.waitKey(0) == 27: break

  print('det time: {0:.4f}'.format(np.mean(runtime['dt'])))









