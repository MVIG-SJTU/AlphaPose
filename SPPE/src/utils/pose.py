from utils.img import (load_image, drawGaussian, drawBigCircle, drawSmallCircle,
                       cropBox, transformBox, transformBoxInvert, flip, shuffleLR, drawCOCO)
from utils.eval import getPrediction
import torch
import numpy as np
import random
import torchsample.transforms as tr
from opt import opt


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.randn(1)[0] * x))


def generateSampleBox(img_path, bndbox, part, nJoints, imgset, scale_factor, dataset, train=True):

    nJoints_coco = 17
    nJoints_mpii = 16
    img = load_image(img_path)
    if train:
        img[0].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
        img[1].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
        img[2].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

    ori_img = img.clone()
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)

    upLeft = torch.Tensor((int(bndbox[0][0]), int(bndbox[0][1])))
    bottomRight = torch.Tensor((int(bndbox[0][2]), int(bndbox[0][3])))
    ht = bottomRight[1] - upLeft[1]
    width = bottomRight[0] - upLeft[0]
    imght = img.shape[1]
    imgwidth = img.shape[2]
    scaleRate = random.uniform(*scale_factor)

    upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
    upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
    bottomRight[0] = min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2)
    bottomRight[1] = min(imght - 1, bottomRight[1] + ht * scaleRate / 2)

    # Doing Random Sample
    if opt.addDPG:
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
            xmin = max(1, min(upLeft[0] + np.random.normal(-0.0142, 0.1158) * width, imgwidth - 3))
            ymin = max(1, min(upLeft[1] + np.random.normal(0.0043, 0.068) * ht, imght - 3))
            xmax = min(max(xmin + 2, bottomRight[0] + np.random.normal(0.0154, 0.1337) * width), imgwidth - 3)
            ymax = min(max(ymin + 2, bottomRight[1] + np.random.normal(-0.0013, 0.0711) * ht), imght - 3)

        upLeft[0] = xmin
        upLeft[1] = ymin
        bottomRight[0] = xmax
        bottomRight[1] = ymax

    # Counting Joints number
    jointNum = 0
    if imgset == 'coco':
        for i in range(17):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
               and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                jointNum += 1
    else:
        for i in range(16):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
               and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                jointNum += 1

    # Doing Random Crop
    if opt.addDPG:
        if jointNum > 13 and train:
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

    ori_inp = cropBox(ori_img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
    inp = cropBox(img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
    if jointNum == 0:
        inp = torch.zeros(3, opt.inputResH, opt.inputResW)

    out_bigcircle = torch.zeros(nJoints, opt.outputResH, opt.outputResW)
    out_smallcircle = torch.zeros(nJoints, opt.outputResH, opt.outputResW)
    out = torch.zeros(nJoints, opt.outputResH, opt.outputResW)
    setMask = torch.zeros(nJoints, opt.outputResH, opt.outputResW)

    # Draw Label
    if imgset == 'coco':
        for i in range(nJoints_coco):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
               and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                out_bigcircle[i] = drawBigCircle(out_bigcircle[i], transformBox(part[i], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss * 2)
                out_smallcircle[i] = drawSmallCircle(out_smallcircle[i], transformBox(part[i], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss)
                out[i] = drawGaussian(out[i], transformBox(part[i], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss)
            setMask[i].add_(1)
    elif imgset == 'mpii':
        for i in range(nJoints_coco, nJoints_coco + nJoints_mpii):
            if part[i - nJoints_coco][0] > 0 and part[i - nJoints_coco][0] > upLeft[0] and part[i - nJoints_coco][1] > upLeft[1] \
               and part[i - nJoints_coco][0] < bottomRight[0] and part[i - nJoints_coco][1] < bottomRight[1]:
                out_bigcircle[i] = drawBigCircle(out_bigcircle[i], transformBox(part[i - nJoints_coco], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss * 2)
                out_smallcircle[i] = drawSmallCircle(out_smallcircle[i], transformBox(part[i - nJoints_coco], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss)
                out[i] = drawGaussian(out[i], transformBox(part[i - nJoints_coco], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss)
            setMask[i].add_(1)
    else:
        for i in range(nJoints_coco, nJoints_coco + nJoints_mpii):
            if part[i - nJoints_coco][0] > 0 and part[i - nJoints_coco][0] > upLeft[0] and part[i - nJoints_coco][1] > upLeft[1] \
               and part[i - nJoints_coco][0] < bottomRight[0] and part[i - nJoints_coco][1] < bottomRight[1]:
                out_bigcircle[i] = drawBigCircle(out_bigcircle[i], transformBox(part[i - nJoints_coco], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss * 2)
                out_smallcircle[i] = drawSmallCircle(out_smallcircle[i], transformBox(part[i - nJoints_coco], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss)
                out[i] = drawGaussian(out[i], transformBox(part[i - nJoints_coco], upLeft, bottomRight, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW), opt.hmGauss)
            if i != 6 + nJoints_coco and i != 7 + nJoints_coco:
                setMask[i].add_(1)

    if opt.debug:
        preds_hm, preds_img, preds_scores = getPrediction(out.unsqueeze(0), upLeft.unsqueeze(0), bottomRight.unsqueeze(0), opt.inputResH,
                                                          opt.inputResW, opt.outputResH, opt.outputResW)
        tmp_preds = preds_hm.mul(opt.inputResH / opt.outputResH)
        drawCOCO(ori_inp.unsqueeze(0), tmp_preds, preds_scores)

    if train:
        # Flip
        if random.uniform(0, 1) < 0.5:
            inp = flip(inp)
            ori_inp = flip(ori_inp)        
            out_bigcircle = shuffleLR(flip(out_bigcircle), dataset)
            out_smallcircle = shuffleLR(flip(out_smallcircle), dataset)
            out = shuffleLR(flip(out), dataset)
        # Rotate
        r = rnd(opt.rotate)
        if random.uniform(0, 1) < 0.6:
            r = 0
        if r != 0:
            rotate = tr.Rotate(r)

            inp = rotate(inp)
            ori_inp = rotate(ori_inp)
            out_bigcircle = rotate(out_bigcircle)
            out_smallcircle = rotate(out_smallcircle)
            out = rotate(out)

    return inp, out_bigcircle, out_smallcircle, out, setMask
