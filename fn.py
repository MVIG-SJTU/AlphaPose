import torch
import re
import os
import collections
from torch._six import string_classes, int_classes
import cv2
from opt import opt
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = True


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def collate_fn_list(batch):
    img, inp, im_name = zip(*batch)
    img = collate_fn(img)
    im_name = collate_fn(im_name)

    return img, inp, im_name


def vis_frame_fast(frame, im_res, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]
    else:
        NotImplementedError

    im_name = im_res['imgname'].split('/')[-1]
    img = frame
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2*(kp_scores[start_p] + kp_scores[end_p]) + 1)
    return img


def vis_frame(frame, im_res, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    im_name = im_res['imgname'].split('/')[-1]
    img = frame
    height,width = img.shape[:2]
    img = cv2.resize(img,(int(width/2), int(height/2)))
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x/2), int(cor_y/2))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x/2), int(cor_y/2)), 2, p_color[n], -1)
            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_scores[n]))
            img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                #cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                transparency = max(0, min(1, 0.5*(kp_scores[start_p] + kp_scores[end_p])))
                img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
    img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
    return img


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval
