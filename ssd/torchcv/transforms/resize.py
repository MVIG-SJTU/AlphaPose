import torch
import random

from PIL import Image


def resize(img, boxes, size, max_size=1000, random_interpolation=False):
    '''Resize the input PIL image to given size.

    If boxes is not None, resize boxes accordingly.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#obj,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
      random_interpolation: (bool) randomly choose a resize interpolation method.

    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.

    Example:
    >> img, boxes = resize(img, boxes, 600)  # resize shorter side to 600
    >> img, boxes = resize(img, boxes, (500,600))  # resize image size to (500,600)
    >> img, _ = resize(img, None, (500,600))  # resize image only
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h

    method = random.choice([
        Image.BOX,
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR]) if random_interpolation else Image.BILINEAR
    img = img.resize((ow,oh), method)
    if boxes is not None:
        boxes = boxes * torch.tensor([sw,sh,sw,sh])
    return img, boxes
