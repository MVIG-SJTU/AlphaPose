import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import WebcamLoader, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import write_json

args = opt
args.dataset = 'coco'


def loop():
    n = 0
    while True:
        yield n
        n += 1

if __name__ == "__main__":
    webcam = args.webcam
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load input video
    fvs = WebcamLoader(webcam).start()
    (fourcc,fps,frameSize) = fvs.videoinfo()
    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_webcam'+webcam+'.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    # Load YOLO model
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_model = Darknet("yolo/cfg/yolov3.cfg")
    det_model.load_weights('models/yolo/yolov3.weights')
    det_model.net_info['height'] = args.inp_dim
    det_inp_dim = int(det_model.net_info['height'])
    assert det_inp_dim % 32 == 0
    assert det_inp_dim > 32
    det_model.cuda()
    det_model.eval()

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

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc =  tqdm(loop())
    for i in im_names_desc:
        try:
            start_time = getTime()

            (img, orig_img, inp, im_dim_list) = fvs.read()
            ckpt_time, load_time = getTime(start_time)
            runtime_profile['ld'].append(load_time)
            with torch.no_grad():
                # Human Detection
                img = Variable(img).cuda()
                im_dim_list = im_dim_list.cuda()

                prediction = det_model(img, CUDA=True)
                ckpt_time, det_time = getTime(ckpt_time)
                runtime_profile['dt'].append(det_time)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                     opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name=str(i)+'.jpg')
                    continue
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5].cpu()
                scores = dets[:, 5:6].cpu()
                ckpt_time, detNMS_time = getTime(ckpt_time)
                runtime_profile['dn'].append(detNMS_time)
                # Pose Estimation
                inps = torch.zeros(boxes.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes.size(0), 2)
                pt2 = torch.zeros(boxes.size(0), 2)
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
                inps = Variable(inps.cuda())

                hm = pose_model(inps)
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)

                writer.save(boxes, scores, hm.cpu(), pt1, pt2, orig_img, im_name=str(i)+'.jpg')
                
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)

            # TQDM
            im_names_desc.set_description(
                'load time: {ld:.4f} | det time: {dt:.4f} | det NMS: {dn:.4f} | pose time: {pt:.4f} | post process: {pn:.4f}'.format(
                    ld=np.mean(runtime_profile['ld']), dt=np.mean(runtime_profile['dt']), dn=np.mean(runtime_profile['dn']),
                    pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )
        except KeyboardInterrupt:
            break

    print(' ')
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)
