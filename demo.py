"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

torch.multiprocessing.set_start_method('forkserver', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

def get_args():
    """----------------------------- Demo options -----------------------------"""
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--sp', default=False, action='store_true')
    parser.add_argument('--detector', dest='detector', default="yolo")
    parser.add_argument('--indir', dest='inputpath', default="")
    parser.add_argument('--list', dest='inputlist', default="")
    parser.add_argument('--image', dest='inputimg', default="")
    parser.add_argument('--outdir', dest='outputpath', default="examples/res/")
    parser.add_argument('--save_img', default=True, action='store_true')
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--profile', default=False, action='store_true')
    parser.add_argument('--format', type=str)
    parser.add_argument('--min_box_area', type=int, default=0)
    parser.add_argument('--detbatch', type=int, default=1)
    parser.add_argument('--posebatch', type=int, default=80)
    parser.add_argument('--eval', dest='eval', default=False, action='store_true')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0")
    parser.add_argument('--qsize', type=int, dest='qsize', default=1024)
    parser.add_argument('--flip', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    """----------------------------- Video options -----------------------------"""
    parser.add_argument('--video', dest='video', default="")
    parser.add_argument('--webcam', dest='webcam', type=int, default=-1)
    parser.add_argument('--save_video', dest='save_video', default=False, action='store_true')
    parser.add_argument('--vis_fast', dest='vis_fast', action='store_true', default=False)
    parser.add_argument('--pose_track', dest='pose_track', action='store_true', default=False)

    args = parser.parse_args([
    '--cfg', './configs/coco/resnet/256x192_res18_lr1e-3_1x.yaml',
    '--checkpoint', './exp/mytrain-res18/final_DPG.pth',
    '--indir', './data/seedland/dense',
    '--outdir', './data/output/'
    ])
    cfg = update_config(args.cfg)

    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    args.detbatch = args.detbatch * len(args.gpus)
    args.posebatch = args.posebatch * len(args.gpus)
    args.tracking = (args.detector == 'tracker')

    return args, cfg


    
def loop():
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":
    args, cfg = get_args()


    mode = 'image'
    for root, dirs, files in os.walk(args.inputpath):
      input_source = files

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    
    detecetor = get_detector(args)

    # Load detection loader
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(input_source, detecetor, cfg, args).start()
    else:
        det_loader = DetectionLoader(input_source, detecetor, cfg, args, batchSize=args.detbatch, mode=mode).start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading pose model from {args.checkpoint}...')
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer, we only takes the image input in this demo
    queueSize = args.qsize
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()
    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], det_loader.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))

                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
        writer.stop()
        det_loader.stop()
    except KeyboardInterrupt:
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues
            writer.commit()
            writer.clear_queues()
            # det_loader.clear_queues()
    final_result = writer.results()
    write_json(final_result, args.outputpath, form=args.format, for_eval=args.eval)
    print("Results have been written to json.")
