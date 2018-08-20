import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import Image_loader, DataWriter, crop_from_dets, Mscoco, DetectionLoader
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction_batch
import os
from tqdm import tqdm
import time
from fn import getTime

from pPose_nms import pose_nms, write_json


args = opt
args.dataset = 'coco'


if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if len(inputlist):
        im_names = open(inputlist, 'r').readlines()
    elif len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')

    # Load input images
    dataset = Image_loader(im_names, format='yolo')

    # Load detection loader
    print('Loading YOLO model..')
    test_loader = DetectionLoader(dataset).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    writer = DataWriter(args.save_video).start()
    time.sleep(10)
    data_len = dataset.__len__()
    im_names_desc = tqdm(range(data_len))
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inp, orig_img, im_name, boxes, scores) = test_loader.read()
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue
            print("test loader:", test_loader.len())
            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation
            inps, pt1, pt2 = crop_from_dets(inp, boxes)
            inps = Variable(inps.cuda())

            hm = pose_model(inps)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            print("writer:" , writer.len())
            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)
        # TQDM
        im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
        )
        '''
        # TQDM
        im_names_desc.set_description(
            'Speed: {speed:.2f}'.format(
                speed=1 / (time.time() - start_time)
            )
        )'''
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)
