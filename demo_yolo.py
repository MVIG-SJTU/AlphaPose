import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import Image_loader, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction_batch
import os
from tqdm import tqdm
import time
from fn import vis_res, getTime

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'


if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load YOLO model
    print('Loading YOLO model..')
    det_model = Darknet("yolo/cfg/yolov3.cfg")
    det_model.load_weights('yolo/yolov3.weights')
    det_model.net_info['height'] = args.inp_dim
    det_inp_dim = int(det_model.net_info['height'])
    assert det_inp_dim % 32 == 0
    assert det_inp_dim > 32
    det_model.cuda()
    det_model.eval()

    print(inputpath)
    print(inputlist)
    if len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    elif len(inputlist):
        with open(inputlist, 'r') as f:
            im_names = []
            for line in f.readlines():
                im_names.append(line.split('\n')[0])
    else:
        raise IOError('Error: ./run.sh must contain either --indir/--list')

    dataset = Image_loader(inputlist, format='yolo')
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=20, pin_memory=True
    )
    im_names_desc = tqdm(test_loader)

    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    #pose_model = torch.nn.DataParallel(pose_model).cuda()
    pose_model.cuda()
    pose_model.eval()

    final_result = []
    runtime_profile = {
        'dt': [],
        'dn': [],
        'pt': [],
        'pn': []
    }
    for i, (img, inp, im_name, im_dim_list) in enumerate(im_names_desc):
        start_time = getTime()
        with torch.no_grad():
            ht = inp.size(2)
            wd = inp.size(3)
            # Human Detection
            img = Variable(img[0]).cuda()
            im_dim_list = im_dim_list[0].cuda()

            prediction = det_model(img, CUDA=True)
            ckpt_time, det_time = getTime(start_time)
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
            inps, pt1, pt2 = crop_from_dets(inp[0], boxes)
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
            # print(len(result))
            result = {
                'imgname': im_name[0],
                'result': result
            }
            final_result.append(result)

        # TQDM
        '''
        im_names_desc.set_description(
            'det time: {dt:.4f} | det NMS: {dn:.4f} | pose time: {pt:.4f} | pose NMS: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), dn=np.mean(runtime_profile['dn']),
                pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
        )'''

        im_names_desc.set_description(
            'Speed: {fps:.2f} FPS'.format(
                fps=1 / (ckpt_time - start_time))
        )
    if not args.vis_res:
        write_json(final_result, args.outputpath)
    else:
        vis_res(final_result, args.outputpath)
