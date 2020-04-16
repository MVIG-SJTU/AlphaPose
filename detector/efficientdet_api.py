import os

import numpy as np

import cv2

import torch

from detector.apis import BaseDetector
from detector.efficientdet.backbone import EfficientDetBackbone
from detector.efficientdet.utils.utils import preprocess, invert_affine, postprocess, BBoxTransform, ClipBoxes





# we shall create a EffDetector here
class EffDetector(BaseDetector):
  input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

  def __init__(self, compound_coef):
    self.model = None
    self.compound_coef = compound_coef
    self.iou_threshold = 0.4
    self.threshold = 0.2
    self.force_input_size = None
    self.input_size = self.input_sizes[self.compound_coef] if self.force_input_size is None else self.force_input_size
    self.regressBoxes = BBoxTransform()
    self.clipBoxes = ClipBoxes()

  def load_model(self):
    self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=90)
    self.model.load_state_dict(torch.load(f'detector/efficientdet/weights/efficientdet-d{self.compound_coef}.pth'))
    self.model.requires_grad_(False)
    self.model.eval()
    self.model = self.model.cuda()
    

  def image_preprocess(self, img_source):
    """
    input:
      image name as a string
    return:
      tensor in (1, 3, h, w) mode
    """
    if isinstance(img_source, str):
      img = cv2.imread(img_source)
    else:
      raise IOError('Unknown image source type: {}'.format(type(img_source)))
    
    ori_imgs, framed_imgs, framed_metas = preprocess(img_source, max_size=self.input_size)
    

    return torch.from_numpy(framed_imgs[0]).unsqueeze(0).permute(0, 3, 1, 2)
    


  def images_detection(self, images, im_dim_list):
    """
    input:
      img tensor in (b, 3, h, w) mode, here b refers to batchsize
      im_dim_list in (b, 4):
        [
          [w, h, w, h],
          [w, h, w, h]
        ]
    return:
      box/score tensor in: (batch_index, [x1, y1, x2, y2], score, class_score(1), class_index(0)) mode
    """
    if self.model is None:
      self.load_model()
    
    # we need to recover framed_metas from im_dim_list
    framed_metas = []
    for dim in im_dim_list:
      old_h, old_w = int(dim[1]), int(dim[0])
      if old_w > old_h:
        new_w = self.input_size
        new_h = int(self.input_size / old_w * old_h)
      else:
        new_w = int(self.input_size / old_h * old_w)
        new_h = self.input_size
      
      padding_h = self.input_size - new_h
      padding_w = self.input_size - new_w

      framed_meta = (new_w, new_h, old_w, old_h, padding_w, padding_h)
      framed_metas.append(framed_meta)
    
    with torch.no_grad():
      torch.cuda.synchronize()
      images = images.cuda()

      features, regression, classification, anchors = self.model(images)

      preds = postprocess(
        images, anchors, regression, classification,
        self.regressBoxes, self.clipBoxes,
        self.threshold, self.iou_threshold
      )

      preds = invert_affine(framed_metas, preds)
    
    # interface out, converting EfficientDet to AlphaPose format input
    results = []
    for idx, pred in enumerate(preds):
      for j, bbox in enumerate(pred['rois']):
        if pred['class_ids'][j] != 0: continue
        x1, y1, x2, y2 = tuple(bbox)
        result = [idx, x1, y1, x2, y2, pred['scores'][j], 1, 0]
        results.append(result)

    return torch.tensor(results)

  def detect_one_img(self, img_name):
    """
    note that framed_metas in (new_w, new_h, old_w, old_h, padding_w, padding_h) mode
    """
    if self.model is None:
      self.load_model()
    ori_imgs, framed_imgs, framed_metas = preprocess(img_name, max_size=self.input_size)

    x = torch.stack([torch.from_numpy(_) for _ in framed_imgs], 0).permute(0, 3, 1, 2)
    x = x.cuda()  # convert to (n, c, h, w)
    
    with torch.no_grad():
      features, regression, classification, anchors = self.model(x)

      preds = postprocess(
        x, anchors, regression, classification,
        self.regressBoxes, self.clipBoxes,
        self.threshold, self.iou_threshold
      )
            
      preds = invert_affine(framed_metas, preds)[0]  # since we only have 1 image here
    
    dets_results = []

    if len(preds['rois']) == 0:
      return dets_results
    
    for i in range(len(preds['rois'])):
      if preds['class_ids'][i] != 0: continue  # we only consider human at this time of view

      # ensamble results
      det_dict = {}
      (x1, y1, x2, y2) = preds['rois'][i]
      x, y, w, h = x1, y1, x2-x1, y2-y1
      det_dict['category_id'] = 1
      det_dict['score'] = float(preds['scores'][i])
      det_dict['bbox'] = [float(_) for _ in [x, y, w, h]]
      try: # only valid using coco dataset
        det_dict['image_id'] = int(os.path.basename(img_name).split('.')[0])
      except:
        # if we are not using coco dataset, we need a random number for image id
        det_dict['image_id'] = np.random.randint(1e6)
      dets_results.append(det_dict)
    
    return dets_results
    