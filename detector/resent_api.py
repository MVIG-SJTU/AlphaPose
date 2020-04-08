import numpy as np

import cv2

import torch

from detector.apis import BaseDetector
from detector.resnet_cfg import cfg
from detector.centorpose.model import create_model, load_model
from detector.centorpose.image import get_affine_transform, transform_preds
from detector.centorpose.utils import flip_lr, flip_lr_off, flip_tensor
from detector.centorpose.decode import _nms, _topk, _transpose_and_gather_feat



class ResDetector(BaseDetector):
  def __init__(self, cfg, opt=None):
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

  def images_detection(self, images, im_dim_list):
    with torch.no_grad():
      torch.cuda.synchronize()
      images = images.cuda()
      hms, whs, hpss, regs, hm_hps, hp_offsets = self.model(images)

    dets = []
    batch_size = len(images)

    for i in range(batch_size):
      hm, wh, hps, reg, hm_hp, hp_offset = hms[i:i+1], whs[i:i+1], hpss[i:i+1], regs[i:i+1], hm_hps[i:i+1], hp_offsets[i:i+1]
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

      # here we have bboxes
      bboxes, scores = bboxes.squeeze(0).detach().cpu(), scores.squeeze(0).detach().cpu()
      bboxes = bboxes[(scores >= cfg.TEST.VIS_THRESH).squeeze(1), :]
      scores = scores[scores >= cfg.TEST.VIS_THRESH].unsqueeze(1)

      dim = im_dim_list[i].numpy()

      # we get exactly the same results with the centerpose
      bboxes = transform_preds(bboxes.reshape(-1, 2), dim[0:2]/2, max(dim[:2]), (128, 128)).reshape(-1, 4)
      bboxes = torch.tensor(bboxes, dtype=scores.dtype)
      batch_ind = bboxes.new_full(scores.shape, i)
      class_score = bboxes.new_full(scores.shape, 1)
      class_index = bboxes.new_full(scores.shape, 0)

      det = torch.cat((batch_ind, bboxes, scores, class_score, class_index), 1)
      dets.append(det)

    return torch.cat(dets)

  def detect_one_img(self, img_name):
    pass
