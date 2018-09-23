# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.data
from predict.annot.coco_minival import Mscoco_minival
from predict.p_poseNMS import pose_nms, write_json
import numpy as np
from predict.opt import opt
from tqdm import tqdm
from utils.img import flip_v, shuffleLR_v, vis_frame
from utils.eval import getPrediction
from utils.eval import getmap
import os
import cv2


def gaussian(size):
    '''
    Generate a 2D gaussian array
    '''
    sigma = 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g = g[np.newaxis, :]
    return g


gaussian_kernel = nn.Conv2d(17, 17, kernel_size=4 * 1 + 1,
                            stride=1, padding=2, groups=17, bias=False)

g = torch.from_numpy(gaussian(4 * 1 + 1)).clone()
g = torch.unsqueeze(g, 1)
g = g.repeat(17, 1, 1, 1)
gaussian_kernel.weight.data = g.float()
gaussian_kernel.cuda()


def prediction(model):
    model.eval()
    dataset = Mscoco_minival()
    minival_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)
    minival_loader_desc = tqdm(minival_loader)

    final_result = []

    tmp_inp = {}
    for i, (inp, box, im_name, metaData) in enumerate(minival_loader_desc):
        #inp = torch.autograd.Variable(inp.cuda(), volatile=True)
        pt1, pt2, ori_inp = metaData
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        if im_name[0] in tmp_inp.keys():
            inps = tmp_inp[im_name[0]]['inps']
            ori_inps = tmp_inp[im_name[0]]['ori_inps']
            boxes = tmp_inp[im_name[0]]['boxes']
            pt1s = tmp_inp[im_name[0]]['pt1s']
            pt2s = tmp_inp[im_name[0]]['pt2s']
            tmp_inp[im_name[0]]['inps'] = torch.cat((inps, inp), dim=0)
            tmp_inp[im_name[0]]['pt1s'] = torch.cat((pt1s, pt1), dim=0)
            tmp_inp[im_name[0]]['pt2s'] = torch.cat((pt2s, pt2), dim=0)
            tmp_inp[im_name[0]]['ori_inps'] = torch.cat(
                (ori_inps, ori_inp), dim=0)
            tmp_inp[im_name[0]]['boxes'] = torch.cat((boxes, box), dim=0)
        else:
            tmp_inp[im_name[0]] = {
                'inps': inp,
                'ori_inps': ori_inp,
                'boxes': box,
                'pt1s': pt1,
                'pt2s': pt2
            }

    for im_name, item in tqdm(tmp_inp.items()):
        inp = item['inps']
        pt1 = item['pt1s']
        pt2 = item['pt2s']
        box = item['boxes']
        ori_inp = item['ori_inps']

        with torch.no_grad():
            try:
                kp_preds = model(inp)
                kp_preds = kp_preds.data[:, :17, :]
            except RuntimeError as e:
                '''
                Divide inputs into two batches
                '''
                assert str(e) == 'CUDA error: out of memory'
                bn = inp.shape[0]
                inp1 = inp[: bn // 2]
                inp2 = inp[bn // 2:]
                kp_preds1 = model(inp1)
                kp_preds2 = model(inp2)
                kp_preds = torch.cat((kp_preds1, kp_preds2), dim=0)
                kp_preds = kp_preds.data[:, :17, :]

            # kp_preds = gaussian_kernel(F.relu(kp_preds))

            # Get predictions
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)

            preds, preds_img, preds_scores = getPrediction(
                kp_preds.cpu().data, pt1, pt2,
                opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW
            )

            result = pose_nms(box, preds_img, preds_scores)
            result = {
                'imgname': im_name,
                'result': result
            }
        #img = display_frame(orig_img, result, opt.outputpath)
        #ori_inp = np.transpose(
        #    ori_inp[0][:3].clone().numpy(), (1, 2, 0)) * 255
        #img = vis_frame(ori_inp, result)
        #cv2.imwrite(os.path.join(
        #    './val', 'vis', im_name), img)
        final_result.append(result)

    write_json(final_result, './val', for_eval=True)
    return getmap()


if __name__ == '__main__':
    prediction()
