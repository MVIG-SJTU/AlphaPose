"""
Integration of centerpose
"""

import time

import numpy as np

import cv2
import torch

from detector.centerpose_api import CenterposeDetector
from detector.centerpose_cfg import cfg





if __name__ == '__main__':
  detector = CenterposeDetector(cfg)
  detector.load_model()

  img_sources = [
    './data/seedland/pose_seg_hard/1.jpg',
    './data/seedland/pose_seg_hard/2.jpg',
    './data/seedland/pose_seg_hard/3.png'
  ]


  imgs = torch.cat([detector.image_preprocess(_) for _ in img_sources])



  orig_imgs = [cv2.imread(_) for _ in img_sources]
  im_dim_list = torch.FloatTensor([(_.shape[1], _.shape[0]) for _ in orig_imgs]).repeat(1, 2)

  print(im_dim_list)
  print(im_dim_list.shape)

  images = imgs.clone()

  # here starts self.images_detection function
  results = detector.images_detection(imgs, im_dim_list)


  # print(detector.detect_one_img(img_sources[0]))
  
  # visualization
  results = results.numpy()
  print(type(results), results)
  for idx in range(orig_imgs.__len__()):
    img = orig_imgs[idx]

    bboxes = results[results[:, 0].astype(np.int) == idx, 1:5]

    for box in bboxes:
      x1, y1, x2, y2 = box.astype(np.int)

      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break


    

