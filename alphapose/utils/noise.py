import math
import numpy as np

import skimage
import cv2

def random_noise(img):
    modes = [
        "gaussian",
        "localvar",
        "poisson",
        "salt",
        "pepper",
        "s&p",
        "speckle",
        None
    ]
    mode = modes[np.random.randint(len(modes))]

    if mode is None:
        return img
    else:
        img_noised = skimage.util.random_noise(img, mode=mode)

        return (img_noised*255).astype(np.uint8)


def random_blur(img):
    modes = [
        'gaussian',
        'median',
        None
    ]
    mode = modes[np.random.randint(len(modes))]
    ksize = 2 * np.random.randint(4) + 1    # random kernel size

    if mode is None:
        return img
    elif mode == 'gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 1)
    elif mode == 'median':
        return cv2.medianBlur(img, ksize)
    else:
        raise 'wrong blur mode!'


def random_aug(img):    
    return random_noise(random_blur(img))
