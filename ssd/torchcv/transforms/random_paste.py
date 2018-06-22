import torch
import random

from PIL import Image


def random_paste(img, boxes, max_ratio=4, fill=0):
    '''Randomly paste the input image on a larger canvas.

    If boxes is not None, adjust boxes accordingly.

    Args:
      img: (PIL.Image) image to be flipped.
      boxes: (tensor) object boxes, sized [#obj,4].
      max_ratio: (int) maximum ratio of expansion.
      fill: (tuple) the RGB value to fill the canvas.

    Returns:
      canvas: (PIL.Image) canvas with image pasted.
      boxes: (tensor) adjusted object boxes.
    '''
    w, h = img.size
    ratio = random.uniform(1, max_ratio)
    ow, oh = int(w*ratio), int(h*ratio)
    canvas = Image.new('RGB', (ow,oh), fill)

    x = random.randint(0, ow - w)
    y = random.randint(0, oh - h)
    canvas.paste(img, (x,y))

    if boxes is not None:
        boxes = boxes + torch.tensor([x,y,x,y], dtype=torch.float)
    return canvas, boxes
