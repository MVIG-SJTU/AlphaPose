'''This random crop strategy is described in paper:
   [1] SSD: Single Shot MultiBox Detector
'''
import math
import torch
import random

from PIL import Image
try:
    from torchcv.utils.box import box_iou, box_clamp
except ImportError:
    from ssd.torchcv.utils.box import box_iou, box_clamp

def random_crop(
        img, boxes, labels,
        min_scale=0.3,
        max_aspect_ratio=2.):
    '''Randomly crop a PIL image.

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      labels: (tensor) bounding box labels, sized [#obj,].
      min_scale: (float) minimal image width/height scale.
      max_aspect_ratio: (float) maximum width/height aspect ratio.

    Returns:
      img: (PIL.Image) cropped image.
      boxes: (tensor) object boxes.
      labels: (tensor) object labels.
    '''
    imw, imh = img.size
    params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
    for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(100):
            scale = random.uniform(min_scale, 1)
            aspect_ratio = random.uniform(
                max(1/max_aspect_ratio, scale*scale),
                min(max_aspect_ratio, 1/(scale*scale)))
            w = int(imw * scale * math.sqrt(aspect_ratio))
            h = int(imh * scale / math.sqrt(aspect_ratio))

            x = random.randrange(imw - w)
            y = random.randrange(imh - h)

            roi = torch.tensor([[x,y,x+w,y+h]], dtype=torch.float)
            ious = box_iou(boxes, roi)
            if ious.min() >= min_iou:
                params.append((x,y,w,h))
                break

    x,y,w,h = random.choice(params)
    img = img.crop((x,y,x+w,y+h))

    center = (boxes[:,:2] + boxes[:,2:]) / 2
    mask = (center[:,0]>=x) & (center[:,0]<=x+w) \
         & (center[:,1]>=y) & (center[:,1]<=y+h)
    if mask.any():
        boxes = boxes[mask] - torch.tensor([x,y,x,y], dtype=torch.float)
        boxes = box_clamp(boxes,0,0,w,h)
        labels = labels[mask]
    else:
        boxes = torch.tensor([[0,0,0,0]], dtype=torch.float)
        labels = torch.tensor([0], dtype=torch.long)
    return img, boxes, labels
