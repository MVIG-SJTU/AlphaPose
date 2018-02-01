#!/usr/bin/env python
 
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect, im_detect_fast
#from model.nms_wrapper import nms
from newnms.nms import  soft_nms
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from tqdm import tqdm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import h5py
 

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'coco':('coco_2014_train+coco_2014_valminusminival',)}

def vis_detections(im, image_name, class_name, dets,xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return num_boxes

    im = im[:, :, (2, 1, 0)]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        num_boxes = num_boxes+1
        results.write("{}\n".format(image_name))
        score_file.write("{}\n".format(score))
        xminarr.append(int(round(bbox[0])));yminarr.append(int(round(bbox[1])));xmaxarr.append(int(round(bbox[2])));ymaxarr.append(int(round(bbox[3])))
        
    return num_boxes

def demo(sess, net, image_name,xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes,imagedir, mode):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(imagedir, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    if mode == 'fast':
        scores, boxes = im_detect_fast(sess, net, im)
    else:    
        scores, boxes = im_detect(sess, net, im)
    # Visualize detections for each class
    CONF_THRESH = 0.1

    # Visualize people
    cls_ind = 1 
    cls = 'person'
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep=soft_nms(dets,method=2)
    dets=keep
    if(dets.shape[0]!=0):
        index_file.write("{} {} ".format(image_name,num_boxes+1))
    num_boxes = vis_detections(im, image_name, cls, dets,xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes, thresh=CONF_THRESH)
    if(dets.shape[0]!=0):
        index_file.write("{}\n".format(num_boxes))
    return num_boxes




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res152')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='coco')
    parser.add_argument('--inputpath', dest='inputpath', help='image-directory', default="")
    parser.add_argument('--inputlist', dest='inputlist', help='image-list', default="")
    parser.add_argument('--mode', dest='mode',help='detection mode, fast/normal/accurate', default=False)
    parser.add_argument('--outputpath', dest='outputpath',help='output-directory', default="")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('../output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    inputpath=args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)
    outputpath=os.path.join(args.outputpath,'BBOX')
    outposepath=os.path.join(args.outputpath,'POSE')
    
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
        os.mkdir(outposepath)
 
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet =='res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 81,
                          tag='default', anchor_scales=[2,4,8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    im_names = []
    print(inputpath)
    print(inputlist)
    if len(inputpath):
        for root,dirs,files in os.walk(inputpath):
            im_names=files
    elif len(inputlist):
        with open(inputlist,'r') as f:
            im_names = []
            for line in f.readlines():
                im_names.append(line.split('\n')[0])
    else:
        raise IOError('Error: ./run.sh must contain either --indir/--list')

    xminarr=[]
    yminarr=[]
    xmaxarr=[]
    ymaxarr=[]
    results = open(os.path.join(outputpath,"test-images.txt"), 'w')
    score_file = open(os.path.join(outputpath,"score-proposals.txt"),'w')
    index_file = open(os.path.join(outputpath,"index.txt"),'w')
 
    num_boxes = 0
    for im_name in tqdm(im_names):
        #print('Human detection for {}'.format(im_name))
        num_boxes=demo(sess, net, im_name, xminarr,yminarr,xmaxarr,ymaxarr,results,score_file,index_file,num_boxes,inputpath, mode)
    with h5py.File(os.path.join(outputpath,"test-bbox.h5"), 'w') as hf:
                    hf.create_dataset('xmin', data=np.array(xminarr))
                    hf.create_dataset('ymin', data=np.array(yminarr))
                    hf.create_dataset('xmax', data=np.array(xmaxarr))
                    hf.create_dataset('ymax', data=np.array(ymaxarr))
    results.close()    
    score_file.close()
    index_file.close()
