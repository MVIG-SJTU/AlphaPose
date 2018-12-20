# coding: utf-8

'''
File: tracker-general.py
Project: AlphaPose
File Created: Tuesday, 18st Dec 2018 14:55:41 pm
-----
Last Modified: Thursday, 20st Dec 2018 23:24:47 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import numpy as np
import os
import json
import copy
import heapq
from munkres import Munkres, print_matrix
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from matching import orb_matching
import argparse

# visualization
def display_pose(imgdir, visdir, tracked, cmap):

    print("Start visualization...\n")
    for imgname in tqdm(tracked.keys()):
        img = Image.open(os.path.join(imgdir,imgname))
        width, height = img.size
        fig = plt.figure(figsize=(width/10,height/10),dpi=10)
        plt.imshow(img)
        for pid in range(len(tracked[imgname])):
            pose = np.array(tracked[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
            tracked_id = tracked[imgname][pid]['idx']

            # keypoint scores of torch version and pytorch version are different
            if np.mean(pose[:,2]) <1 :
                alpha_ratio = 1.0
            else:
                alpha_ratio = 5.0

            if pose.shape[0] == 16:
                mpii_part_names = ['RAnkle','RKnee','RHip','LHip','LKnee','LAnkle','Pelv','Thrx','Neck','Head','RWrist','RElbow','RShoulder','LShoulder','LElbow','LWrist']
                colors = ['m', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'm', 'm', 'r', 'r','b','b']
                pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
                for idx_c, color in enumerate(colors):
                    plt.plot(np.clip(pose[idx_c,0],0,width), np.clip(pose[idx_c,1],0,height), marker='o', 
                            color=color, ms=80/alpha_ratio*np.mean(pose[idx_c,2]), markerfacecolor=(1, 1, 0, 0.7/alpha_ratio*pose[idx_c,2]))
                for idx in range(len(pairs)):
                    plt.plot(np.clip(pose[pairs[idx],0],0,width),np.clip(pose[pairs[idx],1],0,height), 'r-',
                            color=cmap(tracked_id), linewidth=60/alpha_ratio*np.mean(pose[pairs[idx],2]),  alpha=0.6/alpha_ratio*np.mean(pose[pairs[idx],2]))
            elif pose.shape[0] == 17:
                coco_part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
                colors = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
                pairs = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[11,12],[11,13],[13,15],[12,14],[14,16],[6,12],[5,11]]
                for idx_c, color in enumerate(colors):
                    plt.plot(np.clip(pose[idx_c,0],0,width), np.clip(pose[idx_c,1],0,height), marker='o', 
                            color=color, ms=80/alpha_ratio*np.mean(pose[idx_c,2]), markerfacecolor=(1, 1, 0, 0.7/alpha_ratio*pose[idx_c,2]))
                for idx in range(len(pairs)):
                    plt.plot(np.clip(pose[pairs[idx],0],0,width),np.clip(pose[pairs[idx],1],0,height),'r-',
                            color=cmap(tracked_id), linewidth=60/alpha_ratio*np.mean(pose[pairs[idx],2]), alpha=0.6/alpha_ratio*np.mean(pose[pairs[idx],2]))
        plt.axis('off')
        ax = plt.gca()
        ax.set_xlim([0,width])
        ax.set_ylim([height,0])
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if not os.path.exists(visdir): 
            os.mkdir(visdir)
        fig.savefig(os.path.join(visdir,imgname.split()[0]+".png"), pad_inches = 0.0, bbox_inches=extent, dpi=13)
        plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='FoseFlow Tracker')
    parser.add_argument('--imgdir', type=str, required=True, help="Must input the images dir")
    parser.add_argument('--in_json', type=str, required=True, help="result json predicted by AlphaPose")
    parser.add_argument('--out_json', type=str, required=True, help="output path of tracked json")
    parser.add_argument('--visdir', type=str, default="", help="visulization tracked results of video sequences")

    parser.add_argument('--link', type=int, default=100)
    parser.add_argument('--drop', type=float, default=2.0)
    parser.add_argument('--num', type=int, default=7)
    parser.add_argument('--mag', type=int, default=30)
    parser.add_argument('--match', type=float, default=0.2)

    args = parser.parse_args()

    # super parameters
    # 1. look-ahead LINK_LEN frames to find tracked human bbox
    # 2. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score
    # 3. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score(Non DeepMatching)
    # 4. drop low-score(<DROP) keypoints
    # 5. pick high-score(top NUM) keypoints when computing pose_IOU
    # 6. box width/height around keypoint for computing pose IoU
    # 7. match threshold in Hungarian Matching

    link_len = args.link
    weights = [1,2,1,2,0,0] 
    weights_fff = [0,1,0,1,0,0]
    drop = args.drop
    num = args.num
    mag = args.mag
    match_thres = args.match
            
    notrack_json = args.in_json
    tracked_json = args.out_json
    image_dir = args.imgdir
    vis_dir = args.visdir

    # if json format is differnt from "alphapose-forvis.json" (pytorch version)
    if "forvis" not in notrack_json:
        results_forvis = {}
        last_image_name = ' '

        with open(notrack_json) as f:
            results = json.load(f)
            for i in xrange(len(results)):
                imgpath = results[i]['image_id']
                if last_image_name != imgpath:
                    results_forvis[imgpath] = []
                    results_forvis[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
                else:
                    results_forvis[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
                last_image_name = imgpath
        notrack_json = os.path.join(os.path.dirname(notrack_json), "alphapose-results-forvis.json")
        with open(notrack_json,'w') as json_file:
                json_file.write(json.dumps(results_forvis))
        
    notrack = {}
    track = {}
    num_persons = 0

    # load json file without tracking information
    print("Start loading json file...\n")
    with open(notrack_json,'r') as f:
        notrack = json.load(f)
        for img_name in tqdm(sorted(notrack.keys())):
            track[img_name] = {'num_boxes':len(notrack[img_name])}
            for bid in range(len(notrack[img_name])):
                track[img_name][bid+1] = {}
                track[img_name][bid+1]['box_score'] = notrack[img_name][bid]['scores']
                track[img_name][bid+1]['box_pos'] = get_box(notrack[img_name][bid]['keypoints'], os.path.join(image_dir,img_name))
                track[img_name][bid+1]['box_pose_pos'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,0:2]
                track[img_name][bid+1]['box_pose_score'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,-1]
   
    np.save('notrack-bl.npy',track)
    # track = np.load('notrack-bl.npy').item()

    # tracking process
    max_pid_id = 0
    frame_list = sorted(list(track.keys()))

    print("Start pose tracking...\n")
    for idx, frame_name in enumerate(tqdm(frame_list[:-1])):
        frame_new_pids = []
        frame_id = frame_name.split(".")[0]

        next_frame_name = frame_list[idx+1]
        next_frame_id = next_frame_name.split(".")[0]
        
        # init tracking info of the first frame in one video
        if idx == 0:
            for pid in range(1, track[frame_name]['num_boxes']+1):
                    track[frame_name][pid]['new_pid'] = pid
                    track[frame_name][pid]['match_score'] = 0

        max_pid_id = max(max_pid_id, track[frame_name]['num_boxes'])
        cor_file = os.path.join(image_dir, "".join([frame_id, '_', next_frame_id, '_orb.txt']))
       
        # regenerate the missed pair-matching txt
        if not os.path.exists(cor_file) or os.stat(cor_file).st_size<200:
            img1_path = os.path.join(image_dir, frame_name)
            img2_path = os.path.join(image_dir, next_frame_name)
            orb_matching(img1_path,img2_path, image_dir, frame_id, next_frame_id)

        all_cors = np.loadtxt(cor_file)

        # if there is no people in this frame, then copy the info from former frame
        if track[next_frame_name]['num_boxes'] == 0:
            track[next_frame_name] = copy.deepcopy(track[frame_name])
            continue
        cur_all_pids, cur_all_pids_fff = stack_all_pids(track, frame_list[:-1], idx, max_pid_id, link_len)
        match_indexes, match_scores = best_matching_hungarian(
            all_cors, cur_all_pids, cur_all_pids_fff, track[next_frame_name], weights, weights_fff, num, mag)
    
        for pid1, pid2 in match_indexes:
            if match_scores[pid1][pid2] > match_thres:
                track[next_frame_name][pid2+1]['new_pid'] = cur_all_pids[pid1]['new_pid']
                max_pid_id = max(max_pid_id, track[next_frame_name][pid2+1]['new_pid'])
                track[next_frame_name][pid2+1]['match_score'] = match_scores[pid1][pid2]

        # add the untracked new person
        for next_pid in range(1, track[next_frame_name]['num_boxes'] + 1):
            if 'new_pid' not in track[next_frame_name][next_pid]:
                max_pid_id += 1
                track[next_frame_name][next_pid]['new_pid'] = max_pid_id
                track[next_frame_name][next_pid]['match_score'] = 0

    np.save('track-bl.npy',track)
    # track = np.load('track-bl.npy').item()
    
    # calculate number of people
    num_persons = 0
    for fid, frame_name in enumerate(frame_list):
        for pid in range(1, track[frame_name]['num_boxes']+1):
            num_persons = max(num_persons, track[frame_name][pid]['new_pid'])
    print("This video contains %d people."%(num_persons))

    # export tracking result into notrack json files
    print("Export tracking results to json...\n")
    for fid, frame_name in enumerate(tqdm(frame_list)):
        for pid in range(track[frame_name]['num_boxes']):
            notrack[frame_name][pid]['idx'] = track[frame_name][pid+1]['new_pid']

    with open(tracked_json,'w') as json_file:
        json_file.write(json.dumps(notrack))

    if len(args.visdir)>0:
        cmap = plt.cm.get_cmap("hsv", num_persons)
        display_pose(image_dir, vis_dir, notrack, cmap)
