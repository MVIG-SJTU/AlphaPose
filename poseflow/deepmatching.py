# coding: utf-8
'''
File: deepmatching.py
Project: AlphaPose
File Created: Thursday, 1st March 2018 6:05:04 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 1st March 2018 6:18:09 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


import os
from tqdm import tqdm
os.chdir("/deepmatching")
image_dir = "/posetrack_data/images"
imgnames = []
vidnames = []

for a,b,c in os.walk(image_dir):
    if len(a.split("/")) == 9:
        vidnames.append(a)

for vidname in tqdm(sorted(vidnames)):
    # print(vidname)
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
	if not os.path.exists("%s/%s_%s.txt"%(vidname,img1_id,img2_id)):
		cmd = "./deepmatching %s %s -nt 10 -downscale 3 -out %s/%s_%s.txt > cache"%(img1,img2,vidname,img1_id,img2_id)
		os.system(cmd)
