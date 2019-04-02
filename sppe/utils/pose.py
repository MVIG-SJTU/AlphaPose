import random
import scipy.misc

import numpy as np
from opt import opt

from sppe.utils.img import cv_cropBox, cv_rotate, flip, shuffleLR, transformBox, drawGaussian, ToTensor


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.randn(1)[0] * x))


def generateSampleBox(img_path, bndbox, part, nJoints, imgset, scale_factor, dataset, train):
    nJoints_coco = 17
    nJoints_mpii = 16
    # img = ToTensor(image.imread(img_path)).asnumpy()  # RGB
    img = ToTensor(scipy.misc.imread(img_path, mode='RGB')).asnumpy()  # RGB

    if train:  # random color
        img[0] *= (random.uniform(0.7, 1.3))
        img[1] *= (random.uniform(0.7, 1.3))
        img[2] *= (random.uniform(0.7, 1.3))

    img = img.clip(0, 1)
    ori_img = img.copy()

    img[0] -= 0.406
    img[1] -= 0.457
    img[2] -= 0.480

    upLeft = np.array((int(bndbox[0][0]), int(bndbox[0][1])))
    bottomRight = np.array((int(bndbox[0][2]), int(bndbox[0][3])))
    ht = bottomRight[1] - upLeft[1]
    width = bottomRight[0] - upLeft[0]
    imght = img.shape[1]
    imgwidth = img.shape[2]
    scaleRate = random.uniform(*scale_factor)

    upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
    upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
    bottomRight[0] = min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2)
    bottomRight[1] = min(imght - 1, bottomRight[1] + ht * scaleRate / 2)

    inputResH, inputResW = opt.inputResH, opt.inputResW
    outputResH, outputResW = opt.outputResH, opt.outputResW

    # Doing Random Sample
    if opt.addDPG and train:
        PatchScale = random.uniform(0, 1)
        if PatchScale > 0.85:
            ratio = ht / width
            if (width < ht):
                patchWidth = PatchScale * width
                patchHt = patchWidth * ratio
            else:
                patchHt = PatchScale * ht
                patchWidth = patchHt / ratio

            xmin = upLeft[0] + random.uniform(0, 1) * (width - patchWidth)
            ymin = upLeft[1] + random.uniform(0, 1) * (ht - patchHt)
            xmax = xmin + patchWidth + 1
            ymax = ymin + patchHt + 1
        else:
            xmin = max(
                1, min(upLeft[0] + np.random.normal(-0.0142, 0.1158) * width, imgwidth - 3))
            ymin = max(
                1, min(upLeft[1] + np.random.normal(0.0043, 0.068) * ht, imght - 3))
            xmax = min(max(
                xmin + 2, bottomRight[0] + np.random.normal(0.0154, 0.1337) * width), imgwidth - 3)
            ymax = min(
                max(ymin + 2, bottomRight[1] + np.random.normal(-0.0013, 0.0711) * ht), imght - 3)

        upLeft[0] = xmin
        upLeft[1] = ymin
        bottomRight[0] = xmax
        bottomRight[1] = ymax

    upLeft[0] = min(upLeft[0], bottomRight[0] - 5)
    upLeft[1] = min(upLeft[1], bottomRight[0] - 5)
    bottomRight[0] = max(bottomRight[0], upLeft[0] + 5)
    bottomRight[1] = max(bottomRight[1], upLeft[1] + 5)

    joint_num = 0

    if imgset == 'coco':
        for i in range(nJoints_coco):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
               and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                joint_num += 1
    elif imgset == 'mpii':
        for i in range(nJoints_mpii):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
               and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                joint_num += 1
    elif imgset == 'aic':
        for i in range(16):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
               and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                if i != 6 and i != 7:
                    joint_num += 1
    else:
        raise NotImplementedError

    # Doing Random Crop
    if opt.addDPG and train:
        if joint_num > 10:
            switch = random.uniform(0, 1)
            if switch > 0.96:
                bottomRight[0] = (upLeft[0] + bottomRight[0]) / 2
                bottomRight[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.92:
                upLeft[0] = (upLeft[0] + bottomRight[0]) / 2
                bottomRight[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.88:
                upLeft[1] = (upLeft[1] + bottomRight[1]) / 2
                bottomRight[0] = (upLeft[0] + bottomRight[0]) / 2
            elif switch > 0.84:
                upLeft[0] = (upLeft[0] + bottomRight[0]) / 2
                upLeft[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.80:
                bottomRight[0] = (upLeft[0] + bottomRight[0]) / 2
            elif switch > 0.76:
                upLeft[0] = (upLeft[0] + bottomRight[0]) / 2
            elif switch > 0.72:
                bottomRight[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.68:
                upLeft[1] = (upLeft[1] + bottomRight[1]) / 2

    inp = cv_cropBox(img, upLeft, bottomRight, inputResH, inputResW)
    ori_inp = cv_cropBox(ori_img, upLeft, bottomRight, inputResH, inputResW)

    if joint_num == 0:
        inp = np.zeros((3, inputResH, inputResW))

    out = np.zeros((nJoints, outputResH, outputResW), dtype=opt.dtype)
    setMask = np.zeros((nJoints, outputResH, outputResW), dtype=opt.dtype)

    # Draw Label
    if imgset == 'coco':
        for i in range(nJoints_coco):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
               and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                hm_part = transformBox(
                    part[i], upLeft, bottomRight, inputResH, inputResW, outputResH, outputResW)
                out[i] = drawGaussian(out[i], hm_part, opt.hmGauss)

            setMask[i] = 1
    elif imgset == 'mpii':
        for i in range(nJoints_coco, nJoints):
            if part[i - nJoints_coco][0] > 0 and part[i - nJoints_coco][0] > upLeft[0] and part[i - nJoints_coco][1] > upLeft[1] \
               and part[i - nJoints_coco][0] < bottomRight[0] and part[i - nJoints_coco][1] < bottomRight[1]:
                hm_part = transformBox(
                    part[i - nJoints_coco], upLeft, bottomRight, inputResH, inputResW, outputResH, outputResW)
                out[i] = drawGaussian(out[i], hm_part, opt.hmGauss)

            setMask[i] = 1
    elif imgset == 'aic':
        for i in range(nJoints_coco, nJoints):
            if part[i - nJoints_coco][0] > 0 and part[i - nJoints_coco][0] > upLeft[0] and part[i - nJoints_coco][1] > upLeft[1] \
               and part[i - nJoints_coco][0] < bottomRight[0] and part[i - nJoints_coco][1] < bottomRight[1]:
                hm_part = transformBox(
                    part[i - nJoints_coco], upLeft, bottomRight, inputResH, inputResW, outputResH, outputResW)
                out[i] = drawGaussian(out[i], hm_part, opt.hmGauss)

            if i != 6 + nJoints_coco and i != 7 + nJoints_coco:
                setMask[i] = 1

    if train:
        # Flip
        if random.uniform(0, 1) < 0.5:
            inp = flip(inp)
            ori_inp = flip(ori_inp)
            out = shuffleLR(flip(out), dataset)

        # Rotate
        r = rnd(opt.rotate)
        if r != 0 and random.uniform(0, 1) > 0.6:
            inp = cv_rotate(inp, r, opt.inputResW, opt.inputResH)
            ori_inp = cv_rotate(ori_inp, r, opt.inputResW, opt.inputResH)
            out = cv_rotate(out, r, opt.outputResW, opt.outputResH)
    return inp, out, setMask, ori_inp
