import json
import os
import sys
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def getmap(results, file_name='pose-results.json'):
    class Writer(object):
        def __init__(self, stdout):
            self.stdout = stdout

        def write(self, arg):
            pass

        def flush(self):
            self.stdout.flush()

    json_dir = os.path.join('val', file_name)

    with open(json_dir, 'w') as json_file:
        json_file.write(json.dumps(results))

    old_stdout = sys.stdout
    nullwriter = Writer(old_stdout)
    sys.stdout = nullwriter

    try:
        mAP_kp, _ = evaluate_mAP(JsonDir=json_dir, annType='keypoints')
        mAP_det, _ = evaluate_mAP(JsonDir=json_dir, annType='bbox')
    finally:
        sys.stdout = old_stdout

    return mAP_kp, mAP_det


def evaluate_mAP(JsonDir='./val/pose-results.json', annType='keypoints'):
    ListDir = './data/coco/coco-minival500_images.txt'

    # annType = ['segm', 'bbox', 'keypoints']
    print('Running evaluation for *%s* results.' % (annType))

    # load Ground_truth
    annFile = './data/coco/json/person_keypoints_val2014.json'
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
    iouThrs = np.linspace(.5, 0.95, np.round(
        (0.95 - .5) / .05) + 1, endpoint=True)
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
