import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
from opt import opt

from dataloader import Image_loader, crop_from_dets, Mscoco
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction
import os
from tqdm import tqdm
import time
from fn import vis_res
from ssd.torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder
from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'


if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load SSD model
    print('Loading SSD model..')
    det_model = FPNSSD512(num_classes=81).cuda()
    det_model.load_state_dict(torch.load(
        './models/ssd/ssd_coco.pth'))
    det_model.eval()
    box_coder = FPNSSDBoxCoder()

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

    dataset = Image_loader(inputlist)
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

    for i, (img, inp, im_name) in enumerate(im_names_desc):
        start_time = time.time()
        with torch.no_grad():
            ht = inp.size(2)
            wd = inp.size(3)
            # Human Detection
            img = Variable(img).cuda()
            loc_preds, cls_preds = det_model(img)
            if loc_preds.shape[0] == 0:
                continue

            boxes, labels, scores = box_coder.decode(ht, wd,
                loc_preds.data.squeeze().cpu(), F.softmax(cls_preds.squeeze(), dim=1).data.cpu())

            if boxes.shape[0] == 0:
                continue
            assert boxes.shape[0] == scores.shape[0]

            # Pose Estimation
            inps, pt1, pt2 = crop_from_dets(inp[0], boxes)
            inps = Variable(inps.cuda())

            hm = pose_model(inps)

            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            preds_hm, preds_img, preds_scores = getPrediction(
                hm.cpu().data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

            result = pose_nms(boxes, scores, preds_img, preds_scores)
            # print(len(result))
            result = {
                'imgname': im_name[0],
                'result': result
            }
            final_result.append(result)

        # TQDM
        im_names_desc.set_description(
            'Speed: {total:.2f} FPS | Num Poses: {pose}'.format(
                total=1 / (time.time() - start_time),
                pose=len(result['result']))
        )
    if not args.vis_res:
        write_json(final_result, args.outputpath)
    else:
        vis_res(final_result, args.outputpath)
