import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import FileVideoStream, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction_batch
from SPPE.src.utils.img import im_to_torch
import os
from tqdm import tqdm
import time
from fn import vis_frame, display_frame, getTime
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

    # Load input video
    fvs = FileVideoStream(videofile).start()

    # Load YOLO model
    print('Loading YOLO model..')
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

    final_result = []
    runtime_profile = {
        'ld': [],
        'dt': [],
        'dn': [],
        'pt': [],
        'pn': [],
        'vis': []
    }

    im_names_desc =  tqdm(range(fvs.length()))
    for i in im_names_desc:
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
            inps, pt1, pt2 = crop_from_dets(inp, boxes)
            inps = Variable(inps.cuda())

            hm = pose_model(inps)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            preds_hm, preds_img, preds_scores = getPrediction(
                hm.cpu().data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

            result = pose_nms(boxes, scores, preds_img, preds_scores)
            ckpt_time, poseNMS_time = getTime(ckpt_time)
            runtime_profile['pn'].append(poseNMS_time)

            result = {
                'imgname': str(i)+'.jpg',
                'result': result
            }
            final_result.append(result)

        if args.vis_res:
            if not os.path.exists(args.outputpath+'/vis'):
                os.mkdir(args.outputpath+'/vis')
            #display_frame(orig_img, result, args.outputpath)
            vis_frame(orig_img, result, args.outputpath+'/vis')
        ckpt_time, draw_time = getTime(ckpt_time)
        runtime_profile['vis'].append(draw_time)
        # TQDM
        im_names_desc.set_description(
            'load time: {ld:.4f} | det time: {dt:.4f} | det NMS: {dn:.4f} | pose time: {pt:.4f} | pose NMS: {pn:.4f} | visualize: {vis:.4f}'.format(
                ld=np.mean(runtime_profile['ld']), dt=np.mean(runtime_profile['dt']), dn=np.mean(runtime_profile['dn']),
                pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']), vis=np.mean(runtime_profile['vis']))
        )

        # im_names_desc.set_description(
        #     'Speed: {fps:.2f} FPS'.format(
        #         fps=1 / (ckpt_time - start_time))
        # )

    write_json(final_result, args.outputpath, for_eval=False) #set for_eval to True to save the result for COCO server evaluation
