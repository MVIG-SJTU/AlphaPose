# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

from opt import opt
import sys
import numpy as np

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.img import transformBoxInvert


class DataLogger(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


class NullWriter(object):
    def write(self, arg):
        pass


def accuracy(output, label, dataset, out_offset=None):
    if type(output) == list:
        return accuracy(output[opt.nStack - 1], label[opt.nStack - 1], dataset, out_offset)
    else:
        return heatmapAccuracy(output.cpu().data, label.cpu().data, dataset.accIdxs)


def heatmapAccuracy(output, label, idxs):
    preds = getPreds(output)
    gt = getPreds(label)

    norm = torch.ones(preds.size(0)) * opt.outputResH / 10
    dists = calc_dists(preds, gt, norm)

    acc = torch.zeros(len(idxs) + 1)
    avg_acc = 0
    cnt = 0
    for i in range(len(idxs)):
        acc[i + 1] = dist_acc(dists[idxs[i] - 1])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1
    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


def getPreds(hm):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert hm.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hm.view(hm.size(0), hm.size(1), -1), 2)

    maxval = maxval.view(hm.size(0), hm.size(1), 1)
    idx = idx.view(hm.size(0), hm.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hm.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hm.size(3))

    # pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    # preds *= pred_mask
    return preds


def calc_dists(preds, target, normalize):
    preds = preds.float().clone()
    target = target.float().clone()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 0 and target[n, c, 1] > 0:
                dists[c, n] = torch.dist(
                    preds[n, c, :], target[n, c, :]) / normalize[n]
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).float().sum() * 1.0 / dists.ne(-1).float().sum()
    else:
        return - 1


def postprocess(output):
    p = getPreds(output)

    for i in range(p.size(0)):
        for j in range(p.size(1)):
            hm = output[i][j]
            pX, pY = int(round(p[i][j][0])), int(round(p[i][j][1]))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                p[i][j] += diff.sign() * 0.25
    p -= 0.5

    return p


def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(
                round(float(preds[i][j][1])))
            if 1 < pX < opt.outputResW - 2 and 1 < pY < opt.outputResH - 2:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                diff = diff.sign() * 0.25
                diff[1] = diff[1] * inpH / inpW
                preds[i][j] += diff

    preds_tf = torch.zeros(preds.size())
    for i in range(hms.size(0)):        # Number of samples
        for j in range(hms.size(1)):    # Number of output heatmaps for one sample
            preds_tf[i][j] = transformBoxInvert(
                preds[i][j], pt1[i], pt2[i], inpH, inpW, resH, resW)

    return preds, preds_tf, maxval


def getmap(JsonDir='./val/alphapose-results.json'):
    ListDir = '../coco-minival500_images.txt'

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[2]  # specify type here
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    print('Running evaluation for *%s* results.' % (annType))

    # load Ground_truth
    dataType = 'val2014'
    annFile = '../%s_%s.json' % (prefix, dataType)
    cocoGt = COCO(annFile)

    # load Answer(json)
    resFile = JsonDir
    cocoDt = cocoGt.loadRes(resFile)

    # load List
    fin = open(ListDir, 'r')
    imgIds_str = fin.readline()
    if imgIds_str[-1] == '\n':
        imgIds_str = imgIds_str[:-1]
    imgIds_str = imgIds_str.split(',')

    imgIds = []
    for x in imgIds_str:
        imgIds.append(int(x))

    # running evaluation
    iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
    t = np.where(0.5 == iouThrs)[0]

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()

    score = cocoEval.eval['precision'][:, :, :, 0, :]
    mApAll, mAp5 = 0.01, 0.01
    if len(score[score > -1]) != 0:
        score2 = score[t]
        mApAll = np.mean(score[score > -1])
        mAp5 = np.mean(score2[score2 > -1])
    cocoEval.summarize()
    return mApAll, mAp5
