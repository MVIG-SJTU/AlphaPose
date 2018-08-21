import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import VideoDetectionLoader, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'

if __name__ == "__main__":
    videofile = args.video
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)
    
    if not len(videofile):
        raise IOError('Error: must contain --video')

    # Load detection loader
    print('Loading YOLO model..')
    test_loader = VideoDetectionLoader(videofile).start()
    (fourcc,fps,frameSize) = test_loader.videoinfo()

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_'+videofile.split('/')[-1].split('.')[0]+'.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'ld': [],
        'dt': [],
        'dn': [],
        'pt': [],
        'pn': []
    }

    im_names_desc =  tqdm(range(test_loader.length()))
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            # Human Detection
            (inp, orig_img, boxes, scores) = test_loader.read()            
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name=str(i)+'.jpg')
                continue
            #print("test loader:", test_loader.len())
            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)

            # Pose Estimation
            inps, pt1, pt2 = crop_from_dets(inp, boxes)
            inps = Variable(inps.cuda())

            hm = pose_model(inps)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name=str(i)+'.jpg')
            #print("writer:" , writer.len())
            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        # TQDM
        im_names_desc.set_description(
            'det time: {dt:.4f} | pose time: {pt:.4f} | post process: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
        )

    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)
