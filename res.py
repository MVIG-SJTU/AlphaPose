"""
Integration of centerpose
"""

import time

import numpy as np

import cv2
import torch

from detector.apis import BaseDetector
from detector.centorpose.config import cfg, update_config
from detector.centorpose.model import create_model, load_model
from detector.centorpose.image import get_affine_transform
from detector.centorpose.utils import flip_lr, flip_lr_off, flip_tensor
from detector.centorpose.decode import _nms, _topk, _transpose_and_gather_feat


update_config(cfg, './detector/centorpose/config/res_50_512x512.yaml')
cfg.defrost()
cfg.TEST.MODEL_PATH = './detector/centorpose/data/res50_cloud_99.pth'
cfg.DEBUG = 1
cfg.freeze()


class ResDetector(BaseDetector):
  def __init__(self, cfg):
    self.model = None
    self.mean = np.array(cfg.DATASET.MEAN, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(cfg.DATASET.STD, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = cfg.MODEL.NUM_CLASSES
    self.scales = cfg.TEST.TEST_SCALES
    self.cfg = cfg
    self.pause = True
    self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
  
  def load_model(self):
    self.model = create_model('res_50', 64, self.cfg)
    self.model = load_model(self.model, self.cfg.TEST.MODEL_PATH)
    self.model = self.model.to(torch.device('cuda'))
    self.model.eval()

  def image_preprocess(self, img_source):
    """
    sherk: Pre-process the img before fed to the object detection network,
    equals to pre_process in centerpose with fixed scale 1
    """
    if isinstance(img_source, str):
      img = cv2.imread(img_source)
    elif isinstance(img_source, np.ndarray):
      img = img_source
    else:
      raise IOError('Unknown image source type: {}'.format(type(img_source)))

    height, width = img.shape[0:2]
    inp_height, inp_width = self.cfg.MODEL.INPUT_H, self.cfg.MODEL.INPUT_W
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(
      img, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR
    )

    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.cfg.TEST.FLIP_TEST:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)

    images = torch.from_numpy(images)

    return images

  def images_detection(self, images):
    # here starts the @process part
    with torch.no_grad():
      torch.cuda.synchronize()
      
      hm, wh, hps, reg, hm_hp, hp_offset = self.model(images)
      hm = hm.sigmoid_()
      if self.cfg.LOSS.HM_HP and not self.cfg.LOSS.MSE_LOSS:
        hm_hp = hm_hp.sigmoid_()

      reg = reg if self.cfg.LOSS.REG_OFFSET else None
      hm_hp = hm_hp if self.cfg.LOSS.REG_OFFSET else None
      hp_offset = hp_offset if self.cfg.LOSS.REG_HP_OFFSET else None

      if self.cfg.TEST.FLIP_TEST:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        hps = (hps[0:1] + flip_lr_off(hps[1:2], self.flip_idx)) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
          if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
    
    # here starts the @multi_pose_decode part
    K = self.cfg.TEST.TOPK
    batch, cat, height, width = hm.size()
    num_joints = hps.shape[1] // 2
    hm = _nms(hm)
    scores, inds, clses, ys, xs = _topk(hm, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([
      xs - wh[..., 0:1] / 2,
      ys - wh[..., 1:2] / 2,
      xs + wh[..., 0:1] / 2,
      ys + wh[..., 1:2] / 2
    ], dim=2)

    dets = torch.cat([bboxes, scores], dim=2)
    # here starts the @multi_pose_post_process part

    return bboxes, scores

  def detect_one_img(self):
    pass





detector = ResDetector(cfg)
detector.load_model()

imgs = detector.image_preprocess('./data/seedland/pose_seg_hard/1.jpg')

print(imgs.mean())
print(imgs.std())

imgs = imgs.to(torch.device('cuda'))

bboxes, scores = detector.images_detection(imgs)



print(bboxes.mean())
print(scores.mean())

