# coding: utf-8

'''
File: tracker.py
Project: AlphaPose
File Created: Thursday, 1st March 2018 6:12:23 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 1st March 2018 6:17:51 pm
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
from munkres import Munkres, print_matrix
from PIL import Image
from tqdm import tqdm
from utils import *


# posetrack dataset path
image_dir = "/posetrack_data"
# validation annotations path
val_dir = "/posetrack_data/annotations/val"
# pose estimation result filename.json
notrack_json = "alpha-pose-results.json" 
# tracking result dir
track_dir = "val-predict"

if not os.path.exists(track_dir):
        os.mkdir(track_dir)

# super parameters
# 1. look-ahead LINK_LEN frames to find tracked human bbox
# 2. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score
# 3. drop low-score(<DROP) keypoints
# 4. pick high-score(top NUM) keypoints when computing pose_IOU
# 5. box width/height around keypoint for computing pose IoU

link_len = 10 
weights = [0,0,1,0,0,0] 
drop = 2.0 
num = 5  
mag = 50

track = {}
cur_vname = ""
num_persons = 0

# load json file without tracking information
# Note: time is a little long, so it is better to uncomment the following save operation at first time
with open(notrack_json,'r') as f:
    notrack = json.load(f)
    for imgpath in tqdm(sorted(notrack.keys())):
        if 'crop' in imgpath:
            print(imgpath)
        vname,fname = imgpath[:-13],imgpath[-12:]
        if vname != cur_vname:
            cur_vname = vname
            track[vname] = {}
        
        track[vname][fname] = {'num_boxes':len(notrack[imgpath])}
        for bid in range(len(notrack[imgpath])):
            track[vname][fname][bid+1] = {}
            track[vname][fname][bid+1]['box_score'] = notrack[imgpath][bid]['score']
            track[vname][fname][bid+1]['box_pos'] = get_box(notrack[imgpath][bid]['keypoints'], os.path.join(image_dir,imgpath))
            track[vname][fname][bid+1]['box_pose_pos'] = np.array(notrack[imgpath][bid]['keypoints']).reshape(-1,3)[:,0:2]
            track[vname][fname][bid+1]['box_pose_score'] = np.array(notrack[imgpath][bid]['keypoints']).reshape(-1,3)[:,-1]

# np.save('notrack0.1.npy',track)
# track = np.load('notrack0.1.npy').item()

# tracking process
for video_name in tqdm(track.keys()):

    max_pid_id = 0
    frame_list = sorted(list(track[video_name].keys()))

    for idx, frame_name in enumerate(frame_list[:-1]):
        frame_new_pids = []
        frame_id = frame_name.split(".")[0]

        next_frame_name = frame_list[idx+1]
        next_frame_id = next_frame_name.split(".")[0]
        
        if 'crop' in next_frame_name:
            track[video_name][new_frame_name] = copy.deepcopy(track[video_name][frame_name])
            continue
        
        if idx == 0:
            for pid in range(1, track[video_name][frame_name]['num_boxes']+1):
                    track[video_name][frame_name][pid]['new_pid'] = pid
                    track[video_name][frame_name][pid]['match_score'] = 0

        max_pid_id = max(max_pid_id, track[video_name][frame_name]['num_boxes'])
        cor_file = os.path.join(image_dir, video_name, "".join([frame_id, '_', next_frame_id, '.txt']))
        if not os.path.exists(cor_file):
            
            dm = "/home/yuliang/code/PoseTrack-CVPR2017/external/deepmatching/deepmatching"
            img1_path = os.path.join(image_dir,video_name,frame_name)
            img2_path = os.path.join(image_dir,video_name,next_frame_name)
            
            cmd = "%s %s %s -nt 20 -downscale 3 -out %s"%(dm,img1_path,img2_path,cor_file)
            os.system(cmd)
           
        all_cors = np.loadtxt(cor_file)

        if track[video_name][next_frame_name]['num_boxes'] == 0:
            track[video_name][next_frame_name] = copy.deepcopy(track[video_name][frame_name])
            continue
            
        cur_all_pids= stack_all_pids(track[video_name], frame_list[:-1], idx, max_pid_id, link_len)
        match_indexes, match_scores = best_matching_hungarian(
            all_cors, cur_all_pids, track[video_name][next_frame_name], weights, num, mag)
    
        for pid1, pid2 in match_indexes:
            track[video_name][next_frame_name][pid2+1]['new_pid'] = cur_all_pids[pid1]['new_pid']
            max_pid_id = max(max_pid_id, track[video_name][next_frame_name][pid2+1]['new_pid'])
            track[video_name][next_frame_name][pid2+1]['match_score'] = match_scores[pid1][pid2]

        for next_pid in range(1, track[video_name][next_frame_name]['num_boxes'] + 1):
            if 'new_pid' not in track[video_name][next_frame_name][next_pid]:
                max_pid_id += 1
                track[video_name][next_frame_name][next_pid]['new_pid'] = max_pid_id
                track[video_name][next_frame_name][next_pid]['match_score'] = 0

        gap = int(next_frame_id)-int(frame_id)
        if gap>1:
            for i in range(gap):
                if i>0:
                    new_frame_name = "%08d.jpg"%(int(frame_id)+i)
                    track[video_name][new_frame_name] = copy.deepcopy(track[video_name][frame_name])

# np.save('notrack0.2.npy',track)
# track = np.load('notrack0.2.npy').item()
         
rmpe_part_ids = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]

for video_name in tqdm(track.keys()):
    num_persons = 0
    frame_list = sorted(list(track[video_name].keys()))
    for fid, frame_name in enumerate(frame_list):
        for pid in range(1, track[video_name][frame_name]['num_boxes']+1):
            new_score = copy.deepcopy(track[video_name][frame_name][pid]['box_pose_score'])
            new_pose = copy.deepcopy(track[video_name][frame_name][pid]['box_pose_pos'])
            track[video_name][frame_name][pid]['box_pose_score'] = new_score[rmpe_part_ids]
            track[video_name][frame_name][pid]['box_pose_pos'] = new_pose[rmpe_part_ids,:]
            num_persons = max(num_persons, track[video_name][frame_name][pid]['new_pid'])
    track[video_name]['num_persons'] = num_persons

for a,b,c in os.walk(val_dir):
    val_jsons = [item for item in c if 'json' in item]
    break

# export tracking result into json files
for video_name in tqdm(track.keys()):
    name = [item for item in val_jsons if video_name.split("/")[-1] in item]
    if len(name) == 0:
        name = [item for item in val_jsons if video_name.split("/")[-1][1:] in item]
    name = name[0]
    
    final = {'annolist':[]}
    frame_list = list(track[video_name].keys())
    frame_list.remove('num_persons')
    frame_list = sorted(frame_list)
    
    with open(os.path.join(val_dir,name)) as f:
            annot = json.load(f)

    imgs = []
    for img in annot['annolist']:
        imgs.append(img['image'][0]['name'])
            
    for fid, frame_name in enumerate(frame_list):
        if os.path.join(video_name,frame_name) not in imgs:
            continue
        final['annolist'].append({"image":[{"name":os.path.join(video_name,frame_name)}],"annorect":[]})
        for pid in range(1, track[video_name][frame_name]['num_boxes']+1):
            pid_info = track[video_name][frame_name][pid]
            box_pos = pid_info['box_pos']
            box_score = pid_info['box_score']
            pose_pos = pid_info['box_pose_pos']
            pose_score = pid_info['box_pose_score']
            pose_pos = add_nose(pose_pos)
            pose_score = add_nose(pose_score)
            new_pid = pid_info['new_pid']
            
            point_struct = []
            for idx,pose in enumerate(pose_pos):
                if pose_score[idx]>drop:
                    point_struct.append({"id":[idx],"x":[pose[0]],"y":[pose[1]],"score":[pose_score[idx]]})
            final['annolist'][fid]['annorect'].append({"x1":[box_pos[0]],\
                                                        "x2":[box_pos[1]],\
                                                        "y1":[box_pos[2]],\
                                                        "y2":[box_pos[3]],\
                                                        "score":[box_score],\
                                                        "track_id":[new_pid-1],\
                                                        "annopoints":[{"point":point_struct}]})
            
    for rest_name in enumerate(remove_list(imgs,video_name,frame_list)):
        final['annolist'].append({"image":[{"name":rest_name}],"annorect":[]}) 
    with open("%s/%s"%(track_dir,name),'w') as json_file:
        json_file.write(json.dumps(final))
