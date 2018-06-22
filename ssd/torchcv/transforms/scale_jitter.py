import torch
import random

from PIL import Image


def scale_jitter(img, boxes, sizes, max_size=1400):
    '''Randomly scale image shorter side to one of the sizes.

    If boxes is not None, resize boxes accordingly.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#obj,4].
      sizes: (tuple) scale sizes.
      max_size: (int) limit the image longer size to max_size.

    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    size = random.choice(sizes)
    sw = sh = float(size) / size_min
    if sw * size_max > max_size:
        sw = sh = float(max_size) / size_max

    ow = int(w * sw + 0.5)
    oh = int(h * sh + 0.5)
    img = img.resize((ow,oh), Image.BILINEAR)

    if boxes is not None:
        boxes = boxes * torch.tensor([sw,sh,sw,sh])
    return img, boxes
