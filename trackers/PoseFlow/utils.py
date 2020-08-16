# coding: utf-8

'''
File: utils.py
Project: AlphaPose
File Created: Thursday, 1st March 2018 5:32:34 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 20th March 2018 1:18:17 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import numpy as np
import cv2 as cv
import os
import json
import copy
import heapq
from concurrent.futures import ProcessPoolExecutor
from munkres import Munkres, print_matrix
from PIL import Image
from tqdm import tqdm


# keypoint penalty weight
delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891, 0.01402144, \
                    0.03909642, 0.03686941, 0.01981803, 0.03843971, 0.03412318, 0.02415081, \
                    0.01291456, 0.01236173,0.01291456, 0.01236173])


# get expand bbox surrounding single person's keypoints
def get_box(pose, imgpath):

    pose = np.array(pose).reshape(-1,3)
    xmin = np.min(pose[:,0])
    xmax = np.max(pose[:,0])
    ymin = np.min(pose[:,1])
    ymax = np.max(pose[:,1])
    
    img_height, img_width, _ = cv.imread(imgpath).shape

    return expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height)

# expand bbox for containing more background
def expand_bbox(left, right, top, bottom, img_width, img_height):

    width = right - left
    height = bottom - top
    ratio = 0.1 # expand ratio
    new_left = np.clip(left - ratio * width, 0, img_width)
    new_right = np.clip(right + ratio * width, 0, img_width)
    new_top = np.clip(top - ratio * height, 0, img_height)
    new_bottom = np.clip(bottom + ratio * height, 0, img_height)

    return [int(new_left), int(new_right), int(new_top), int(new_bottom)]

# calculate final matching grade
def cal_grade(l, w):
    return sum(np.array(l)*np.array(w))

# calculate IoU of two boxes(thanks @ZongweiZhou1)
def cal_bbox_iou(boxA, boxB): 

    xA = max(boxA[0], boxB[0]) #xmin
    yA = max(boxA[2], boxB[2]) #ymin
    xB = min(boxA[1], boxB[1]) #xmax
    yB = min(boxA[3], boxB[3]) #ymax

    if xA < xB and yA < yB: 
        interArea = (xB - xA + 1) * (yB - yA + 1) 
        boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1) 
        boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1) 
        iou = interArea / float(boxAArea + boxBArea - interArea+0.00001) 
    else: 
        iou=0.0

    return iou

# calculate OKS between two single poses
def compute_oks(anno, predict, delta):
    
    xmax = np.max(np.vstack((anno[:, 0], predict[:, 0])))
    xmin = np.min(np.vstack((anno[:, 0], predict[:, 0])))
    ymax = np.max(np.vstack((anno[:, 1], predict[:, 1])))
    ymin = np.min(np.vstack((anno[:, 1], predict[:, 1])))
    scale = (xmax - xmin) * (ymax - ymin)
    dis = np.sum((anno - predict)**2, axis=1)
    oks = np.mean(np.exp(-dis / 2 / delta**2 / scale))

    return oks

# stack all already tracked people's info together(thanks @ZongweiZhou1)
def stack_all_pids(track_vid, frame_list, idxs, max_pid_id, link_len):
    
    #track_vid contains track_vid[<=idx]
    all_pids_info = []
    all_pids_fff = [] # boolean list, 'fff' means From Former Frame
    all_pids_ids = [(item+1) for item in range(max_pid_id)]
    
    for idx in np.arange(idxs,max(idxs-link_len,-1),-1):
        for pid in range(1, track_vid[frame_list[idx]]['num_boxes']+1):
            if len(all_pids_ids) == 0:
                return all_pids_info, all_pids_fff
            elif track_vid[frame_list[idx]][pid]['new_pid'] in all_pids_ids:
                all_pids_ids.remove(track_vid[frame_list[idx]][pid]['new_pid'])
                all_pids_info.append(track_vid[frame_list[idx]][pid])
                if idx == idxs:
                    all_pids_fff.append(True)
                else:
                    all_pids_fff.append(False)
    return all_pids_info, all_pids_fff

# calculate DeepMatching Pose IoU given two boxes
def find_two_pose_box_iou(pose1_box, pose2_box, all_cors):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = pose1_box
    x1_region_ids = set(np.where((x1 >= x_min) & (x1 <= x_max))[0].tolist())
    y1_region_ids = set(np.where((y1 >= y_min) & (y1 <= y_max))[0].tolist())
    region_ids1 = x1_region_ids & y1_region_ids
    x_min, x_max, y_min, y_max = pose2_box
    x2_region_ids = set(np.where((x2 >= x_min) & (x2 <= x_max))[0].tolist())
    y2_region_ids = set(np.where((y2 >= y_min) & (y2 <= y_max))[0].tolist())
    region_ids2 = x2_region_ids & y2_region_ids
    inter = region_ids1 & region_ids2
    union = region_ids1 | region_ids2
    pose_box_iou = len(inter) / (len(union) + 0.00001)

    return pose_box_iou

# calculate general Pose IoU(only consider top NUM matched keypoints)
def cal_pose_iou(pose1_box,pose2_box, num,mag):
    
    pose_iou = []
    for row in range(len(pose1_box)):
        x1,y1 = pose1_box[row]
        x2,y2 = pose2_box[row]
        box1 = [x1-mag,x1+mag,y1-mag,y1+mag]
        box2 = [x2-mag,x2+mag,y2-mag,y2+mag]
        pose_iou.append(cal_bbox_iou(box1,box2))

    return np.mean(heapq.nlargest(num, pose_iou))

# calculate DeepMatching based Pose IoU(only consider top NUM matched keypoints)
def cal_pose_iou_dm(all_cors,pose1,pose2,num,mag):
    
    poses_iou = []
    for ids in range(len(pose1)):
        pose1_box = [pose1[ids][0]-mag,pose1[ids][0]+mag,pose1[ids][1]-mag,pose1[ids][1]+mag]
        pose2_box = [pose2[ids][0]-mag,pose2[ids][0]+mag,pose2[ids][1]-mag,pose2[ids][1]+mag]
        poses_iou.append(find_two_pose_box_iou(pose1_box, pose2_box, all_cors))

    return np.mean(heapq.nlargest(num, poses_iou))
        
# hungarian matching algorithm(thanks @ZongweiZhou1)
def _best_matching_hungarian(all_cors, all_pids_info, all_pids_fff, track_vid_next_fid, weights, weights_fff, num, mag):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    all_grades_details = []
    all_grades = []
    
    box1_num = len(all_pids_info)
    box2_num = track_vid_next_fid['num_boxes']
    cost_matrix = np.zeros((box1_num, box2_num))

    for pid1 in range(box1_num):
        box1_pos = all_pids_info[pid1]['box_pos']
        box1_region_ids = find_region_cors_last(box1_pos, all_cors)
        box1_score = all_pids_info[pid1]['box_score']
        box1_pose = all_pids_info[pid1]['box_pose_pos']
        box1_fff = all_pids_fff[pid1]

        for pid2 in range(1, track_vid_next_fid['num_boxes'] + 1):
            box2_pos = track_vid_next_fid[pid2]['box_pos']
            box2_region_ids = find_region_cors_next(box2_pos, all_cors)
            box2_score = track_vid_next_fid[pid2]['box_score']
            box2_pose = track_vid_next_fid[pid2]['box_pose_pos']
                        
            inter = box1_region_ids & box2_region_ids
            union = box1_region_ids | box2_region_ids
            dm_iou = len(inter) / (len(union) + 0.00001)
            box_iou = cal_bbox_iou(box1_pos, box2_pos)
            pose_iou_dm = cal_pose_iou_dm(all_cors, box1_pose, box2_pose, num,mag)
            pose_iou = cal_pose_iou(box1_pose, box2_pose,num,mag)
            if box1_fff:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights)
            else:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights_fff)
                
            cost_matrix[pid1, pid2 - 1] = grade
    m = Munkres()
    indexes = m.compute((-np.array(cost_matrix)).tolist())

    return indexes, cost_matrix

# multiprocessing version of hungarian matching algorithm
def best_matching_hungarian(all_cors, all_pids_info, all_pids_fff, track_vid_next_fid, weights, weights_fff, num, mag, pool_size=5):
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    all_grades_details = []
    all_grades = []
    
    box1_num = len(all_pids_info)
    box2_num = track_vid_next_fid['num_boxes']
    cost_matrix = np.zeros((box1_num, box2_num))

    qsize = box1_num * track_vid_next_fid['num_boxes']
    pool = ProcessPoolExecutor(max_workers=pool_size)
    futures = []
    for pid1 in range(box1_num):
        box1_pos = all_pids_info[pid1]['box_pos']
        box1_region_ids = find_region_cors_last(box1_pos, all_cors)
        box1_score = all_pids_info[pid1]['box_score']
        box1_pose = all_pids_info[pid1]['box_pose_pos']
        box1_fff = all_pids_fff[pid1]

        for pid2 in range(1, track_vid_next_fid['num_boxes'] + 1):
            future = pool.submit(best_matching_hungarian_kernel, pid1, pid2, all_cors, track_vid_next_fid, weights, weights_fff, num, mag, box1_pos, box1_region_ids, box1_score, box1_pose, box1_fff)
            futures.append(future)

    pool.shutdown(True)
    for future in futures:
        pid1, pid2, grade = future.result()
        cost_matrix[pid1, pid2 - 1] = grade
    m = Munkres()
    indexes = m.compute((-np.array(cost_matrix)).tolist())

    return indexes, cost_matrix

# one iteration of hungarian matching algorithm
def best_matching_hungarian_kernel(pid1, pid2, all_cors, track_vid_next_fid, weights, weights_fff, num, mag, box1_pos, box1_region_ids, box1_score, box1_pose, box1_fff):
    box2_pos = track_vid_next_fid[pid2]['box_pos']
    box2_region_ids = find_region_cors_next(box2_pos, all_cors)
    box2_score = track_vid_next_fid[pid2]['box_score']
    box2_pose = track_vid_next_fid[pid2]['box_pose_pos']
                        
    inter = box1_region_ids & box2_region_ids
    union = box1_region_ids | box2_region_ids
    dm_iou = len(inter) / (len(union) + 0.00001)
    box_iou = cal_bbox_iou(box1_pos, box2_pos)
    pose_iou_dm = cal_pose_iou_dm(all_cors, box1_pose, box2_pose, num,mag)
    pose_iou = cal_pose_iou(box1_pose, box2_pose,num,mag)
    if box1_fff:
        grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights)
    else:
        grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights_fff)  
    return (pid1, pid2, grade)

# calculate number of matching points in one box from last frame
def find_region_cors_last(box_pos, all_cors):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = box_pos
    x1_region_ids = set(np.where((x1 >= x_min) & (x1 <= x_max))[0].tolist())
    y1_region_ids = set(np.where((y1 >= y_min) & (y1 <= y_max))[0].tolist())
    region_ids = x1_region_ids & y1_region_ids

    return region_ids

# calculate number of matching points in one box from next frame
def find_region_cors_next(box_pos, all_cors):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = box_pos
    x2_region_ids = set(np.where((x2 >= x_min) & (x2 <= x_max))[0].tolist())
    y2_region_ids = set(np.where((y2 >= y_min) & (y2 <= y_max))[0].tolist())
    region_ids = x2_region_ids & y2_region_ids

    return region_ids

# fill the nose keypoint by averaging head and neck
def add_nose(array):
    
    if min(array.shape) == 2:
        head = array[-1,:]
        neck = array[-2,:]
    else:
        head = array[-1]
        neck = array[-2]
    nose = (head+neck)/2.0

    return np.insert(array,-1,nose,axis=0)

# list remove operation
def remove_list(l1,vname,l2):
    
    for item in l2:
        l1.remove(os.path.join(vname,item))
        
    return l1
