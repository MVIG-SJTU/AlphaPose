# coding: utf-8
'''
File: matching.py
Project: AlphaPose
File Created: Monday, 1st October 2018 12:53:12 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


import os
import cv2
from tqdm import tqdm
import numpy as np
import time
import argparse

def generate_fake_cor(img, out_path):
    print("Generate fake correspondence files...%s"%out_path)
    fd = open(out_path,"w")
    height, width, channels = img.shape

    for x in range(width):
        for y in range(height):
            ret = fd.write("%d %d %d %d %f \n"%(x, y, x, y, 1.0))
    fd.close()


def orb_matching(img1_path, img2_path, vidname, img1_id, img2_id):
    
    out_path = "%s/%s_%s_orb.txt"%(vidname, img1_id, img2_id)
    # print(out_path)
    
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
    
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    if len(kp1)*len(kp2) < 400:
        generate_fake_cor(img1, out_path)
        return

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 12, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 2) #2

    search_params = dict(checks=100)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    
    # Open file
    fd = open(out_path,"w")

    # ratio test as per Lowe's paper
    for i, m_n in enumerate(matches):
        if len(m_n) != 2:
            continue
        elif m_n[0].distance < 0.80*m_n[1].distance:
            ret = fd.write("%d %d %d %d %f \n"%(kp1[m_n[0].queryIdx].pt[0], kp1[m_n[0].queryIdx].pt[1], kp2[m_n[0].trainIdx].pt[0], kp2[m_n[0].trainIdx].pt[1], m_n[0].distance))
    
    # Close opened file
    fd.close()

    # print(os.stat(out_path).st_size)

    if os.stat(out_path).st_size<1000:
        generate_fake_cor(img1, out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='FoseFlow Matching')
    parser.add_argument('--orb', type=int, default=0)
    args = parser.parse_args()

    image_dir = "posetrack_data/images"
    imgnames = []
    vidnames = []

    for a,b,c in os.walk(image_dir):
        if len(a.split("/")) == 4:
            vidnames.append(a)

    for vidname in tqdm(sorted(vidnames)):
        for a,b,c in os.walk(vidname):
            c=[item for item in c if "jpg" in item]
            imgnames = sorted(c)
            break
        for imgname in imgnames[:-1]:
            if 'crop' in imgname:
                continue
            img1 = os.path.join(vidname,imgname)
            len_name = len(imgname.split(".")[0])
            if len_name == 5:
                img2 = os.path.join(vidname,"%05d.jpg"%(int(imgname.split(".")[0])+1))
            else:
                img2 = os.path.join(vidname,"%08d.jpg"%(int(imgname.split(".")[0])+1))
            if not os.path.exists(img2):
                continue
            img1_id = img1.split(".")[0].split("/")[-1]
            img2_id = img2.split(".")[0].split("/")[-1]
            if args.orb:
                cor_file = "%s/%s_%s_orb.txt"%(vidname,img1_id,img2_id)
            else:
                cor_file = "%s/%s_%s.txt"%(vidname,img1_id,img2_id)
            if not os.path.exists(cor_file) or os.stat(cor_file).st_size<1000:
                if args.orb:
                    # calc orb matching
                    orb_matching(img1,img2,vidname,img1_id,img2_id)
                else:
                    # calc deep matching
                    cmd = "./deepmatching/deepmatching %s %s -nt 10 -downscale 3 -out %s/%s_%s.txt > cache"%(img1,img2,vidname,img1_id,img2_id)
                    os.system(cmd)
