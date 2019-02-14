import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import torch.utils.data as data

import os
import sys
import cv2
import json
import time
import numpy as np

from opt import opt
from tqdm import tqdm

from yolo.preprocess import prep_image
from yolo.darknet import Darknet
from yolo.util import dynamic_write_results

from dataloader import Mscoco, crop_from_dets
from SPPE.src.utils.eval import getPrediction
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from SPPE.src.main_fast_inference import InferenNet
from pPose_nms import pose_nms, write_json
from fn import vis_frame

def load_models():
    font = cv2.FONT_HERSHEY_SIMPLEX

    body2parts = {0: "Nose", 1:"L_Eye", 2: "R_Eye", 3:"L_Ear", 4:"R_Ear",
                  5:"LShoulder", 6:"RShoulder", 7:"LElbow", 8: "RElbow", 9:"LWrist", 10:"RWrist",
                  11: "LHip", 12: "RHip", 13: "LKnee", 14:"RKnee", 15:"LAnkle", 16:"RAnkle", 17:"Neck"}

    part2body = {val:key for (key, val) in body2parts.items()}
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights('models/yolo/yolov3-spp.weights')
    det_model.net_info['height'] = opt.inp_dim
    det_inp_dim = int(det_model.net_info['height'])

    det_model.cuda()
    det_model.eval()

    # Load pose model
    pose_dataset = Mscoco()
    pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    return det_model, pose_model

def predict_single_image(im_name, det_model, pose_model):

    det_inp_dim = int(det_model.net_info['height'])
    inp_dim = int(opt.inp_dim)
    img, orig_img, im_dim = prep_image(im_name, int(opt.inp_dim))
    im_dim = torch.FloatTensor(im_dim).repeat(1,2)

    with torch.no_grad():

        im_dim = torch.FloatTensor(im_dim).repeat(1,2)    
        # Human Detection
        img = img.cuda()
        prediction = det_model(img, CUDA=True)

        
        # NMS process
        dets = dynamic_write_results(prediction, opt.confidence,
                            opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
        dets = dets.cpu()
        im_dim_list = torch.index_select(im_dim, 0, dets[:, 0].long())
        scaling_factor = torch.min(det_inp_dim / im_dim, 1)[0].view(-1, 1)

        # coordinate transfer
        dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2


        dets[:, 1:5] /= scaling_factor
        for j in range(dets.shape[0]):
            dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
            dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]


        boxes = boxes[dets[:,0]==0]

        inps = torch.zeros(boxes.size(0), 3, opt.inputResH, opt.inputResW)
        pt1 = torch.zeros(boxes.size(0), 2)
        pt2 = torch.zeros(boxes.size(0), 2)
        scores = scores[dets[:,0]==0]

        inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

        inps = inps.cuda()
        hm = pose_model(inps)
        hm = hm.cpu()


    if boxes is not None:
        preds_hm, preds_img, preds_scores = getPrediction(hm, pt1, pt2, 
                                                          opt.inputResH, opt.inputResW,
                                                          opt.outputResH, opt.outputResW)
        result = pose_nms(boxes, scores, preds_img, preds_scores)

        result = {
            'imgname': im_name,
            'result': result
        }
        
        return result, orig_img


imagePath = "mustafa/2.jpg"

det_model, pose_model = load_models()
result, orig_img = predict_single_image(imagePath, det_model, pose_model)

img = vis_frame(orig_img, result)
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
