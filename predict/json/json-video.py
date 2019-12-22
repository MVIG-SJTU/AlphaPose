# coding: utf-8

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
import shutil
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--outputpath',dest='outputpath', help='path of output', default="")
    parser.add_argument('--inputpath',dest='inputpath', help='path of inputpath', default="")
    args = parser.parse_args()
    return args

def display_pose(intputpath, outputpath, imgname):
    img = Image.open(os.path.join(intputpath,imgname))
    width, height = img.size
    fig = plt.figure(figsize=(width/10,height/10),dpi=10)
    plt.imshow(img)
    for pid in range(len(rmpe_results[imgname])):
        pose = np.array(rmpe_results[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
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
    fig.savefig(os.path.join(outputpath,'RENDER',imgname.split('.')[0]+'.png'),pad_inches = 0.0, bbox_inches=extent, dpi=13)

    #fig.savefig(os.path.join(outputpath,'RENDER',imgname.split('/')[-1]),pad_inches = 0.0, bbox_inches=extent, dpi=13)
    plt.close()

        
if __name__ == '__main__':
    args = parse_args()
    outputpath = args.outputpath
    inputpath = args.inputpath
    jsonpath = os.path.join(args.outputpath,"POSE/alpha-pose-results-forvis.json")
    
    with open(jsonpath) as f:
        rmpe_results = json.load(f)
    for imgname in tqdm(rmpe_results.keys()):
        display_pose(inputpath, outputpath, imgname)    
