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

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)

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

def display_pose(final_result, outputpath):
    img = cv2.imread(os.path.join(opt.inputpath, im_name))
    width, height = img.size
    fig = plt.figure(figsize=(width/10,height/10),dpi=10)
    plt.imshow(img)
    for pid in range(len(final_result[imgname])):
        pose = final_result[imgname][pid]['keypoints']
        kp_scores = human['kp_score']
        if pose.shape[0] == 16:
            mpii_part_names = ['RAnkle','RKnee','RHip','LHip','LKnee','LAnkle','Pelv','Thrx','Neck','Head','RWrist','RElbow','RShoulder','LShoulder','LElbow','LWrist']
            colors = ['m', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'm', 'm', 'r', 'r','b','b']
            pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
            colors_skeleton = ['m', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'm', 'r', 'r', 'b','b']
            for idx_c, color in enumerate(colors):
                plt.plot(np.clip(pose[idx_c,0],0,width), np.clip(pose[idx_c,1],0,height), marker='o', color=color, ms=40*np.mean(pose[idx_c,2]))
            for idx in range(len(colors_skeleton)):
                plt.plot(np.clip(pose[pairs[idx],0],0,width),np.clip(pose[pairs[idx],1],0,height), 'r-',
                        color=colors_skeleton[idx],linewidth=40*np.mean(pose[pairs[idx],2]),  alpha=np.mean(pose[pairs[idx],2]))
        elif pose.shape[0] == 17:
            coco_part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
            colors = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
            pairs = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[11,12],[11,13],[13,15],[12,14],[14,16],[6,12],[5,11]]
            colors_skeleton = ['y', 'y', 'y', 'y', 'b', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r','m','m']
            for idx_c, color in enumerate(colors):
                plt.plot(np.clip(pose[idx_c,0],0,width), np.clip(pose[idx_c,1],0,height), marker='o', color=color, ms=4*np.mean(pose[idx_c,2]))
            for idx in range(len(colors_skeleton)):
                plt.plot(np.clip(pose[pairs[idx],0],0,width),np.clip(pose[pairs[idx],1],0,height),'r-',
                         color=colors_skeleton[idx],linewidth=4*np.mean(pose[pairs[idx],2]), alpha=0.12*np.mean(pose[pairs[idx],2]))

    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim([0,width])
    ax.set_ylim([height,0])
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(outputpath,'vis',imgname.split('/')[-1]),pad_inches = 0.0, bbox_inches=extent, dpi=13)
    plt.close()

def vis_res(final_result, outputpath, format='coco'):
    '''
    final_result: result dict of predictions
    outputpath: output directory
    format: coco or mpii
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                   YELLOW, ORANGE, PINK, RED, PINK, RED, PINK, RED]
    else:
        NotImplementedError

    for im_res in (final_result):
        im_name = im_res['imgname'].split('/')[-1]
        img = cv2.imread(os.path.join(opt.inputpath, im_name))
        for human in im_res['result']:
            part_line = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            # Draw keypoints
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= 0.3:
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                part_line[n] = (cor_x, cor_y)
                cv2.circle(img, (cor_x, cor_y), 5, p_color[n], -1)
            # Draw limbs
            for start_p, end_p in l_pair:
                if start_p in part_line and end_p in part_line:
                    start_p = part_line[start_p]
                    end_p = part_line[end_p]
                    cv2.line(img, start_p, end_p, YELLOW, 2)

        cv2.imwrite(os.path.join(outputpath, im_name), img)

def display_frame(frame, im_res, outputpath, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    outputpath: output directory
    format: coco or mpii
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
        line_color =  ['y', 'y', 'y', 'y', 'b', 'b', 'b', 'b', 'b', 'm', 'm', 'r', 'r', 'r','r']
    elif format == 'mpii':
        l_pair = [
            (8,9),(11,12),(11,10),(2,1),(1,0),
            (13,14),(14,15),(3,4),(4,5),
            (8,7),(7,6),(6,2),(6,3),(8,12),(8,13)
        ]
        p_color = ['m', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'm', 'm', 'r', 'r','b','b']
        line_color = ['m', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'm', 'r', 'r', 'b','b']
    else:
        NotImplementedError

    img = Image.fromarray(frame)
    width, height = img.size
    fig = plt.figure(figsize=(width/10,height/10),dpi=10)
    plt.imshow(img)
    imgname = im_res['imgname'].split('/')[-1]
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            plt.plot(np.clip(cor_x,0,width), np.clip(cor_y,0,height), marker='o', color=p_color[n], ms=10*kp_scores[n])
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                plt.plot(np.clip((start_xy[0],end_xy[0]),0,width),np.clip((start_xy[1],end_xy[1]),0,height), 'r-',
                        color=line_color[i],linewidth=2*(kp_scores[start_p]+kp_scores[end_p]),  alpha=0.06*(kp_scores[start_p]+kp_scores[end_p]))
    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim([0,width])
    ax.set_ylim([height,0])
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(outputpath,'vis',imgname.split('/')[-1]),pad_inches = 0.0, bbox_inches=extent, dpi=13)
    plt.close()

def vis_frame(frame, im_res, outputpath, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    outputpath: output directory
    format: coco or mpii
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                   YELLOW, ORANGE, PINK, RED, PINK, RED, PINK, RED]
    else:
        NotImplementedError

    im_name = im_res['imgname'].split('/')[-1]
    img = frame
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.3:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 5, p_color[n], -1)
        # Draw limbs
        for start_p, end_p in l_pair:
            if start_p in part_line and end_p in part_line:
                start_p = part_line[start_p]
                end_p = part_line[end_p]
                cv2.line(img, start_p, end_p, YELLOW, 2)

    cv2.imwrite(os.path.join(outputpath, im_name), img)

def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval
