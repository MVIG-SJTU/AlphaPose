import torch
import random

from PIL import Image


def random_flip(img, boxes):
    '''Randomly flip PIL image.

    If boxes is not None, flip boxes accordingly.

    Args:
      img: (PIL.Image) image to be flipped.
      boxes: (tensor) object boxes, sized [#obj,4].

    Returns:
      img: (PIL.Image) randomly flipped image.
      boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        if boxes is not None:
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
    return img, boxes
