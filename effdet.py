from glob import glob
import json

import numpy as np

import cv2

import torch

from detector.efficientdet_api import EffDetector

def demo_more(detector, img_sources):
  imgs = torch.cat([detector.image_preprocess(_) for _ in img_sources])
  orig_imgs = [cv2.imread(_) for _ in img_sources]
  im_dim_list = torch.FloatTensor([(_.shape[1], _.shape[0]) for _ in orig_imgs]).repeat(1, 2)


  results = detector.images_detection(imgs, im_dim_list)

  # visualization
  results = results.numpy()
  for idx, img in enumerate(orig_imgs):
    bboxes = results[results[:, 0].astype(np.int) == idx, 1:5]
    scores = results[results[:, 0].astype(np.int) == idx, 5]

    for j, box in enumerate(bboxes):
      x1, y1, x2, y2 = box.astype(np.int)
      score_txt = '{:.3f}'.format(scores[j])

      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(img, score_txt, (x1, y1+10), 0, 0.75, (0, 255, 0), 2)

    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break



if __name__ == "__main__":
  img_sources = glob('./data/seedland/testpng/*')[:1]

  detector = EffDetector(4)
  detector.load_model()

  dets = []

  for source in img_sources:
    det = detector.detect_one_img(source)
    if det:
      dets += det

  json.dump(dets, open('dummy.json', 'w'))





