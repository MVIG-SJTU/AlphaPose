import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import WebcamLoader, ImageLoader, VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import os, sys, ntpath
import platform
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'

curPlatform = platform.system()

if curPlatform == 'Windows':
  args.sp = True

if not args.sp:
  torch.multiprocessing.set_start_method('forkserver', force=True)
  torch.multiprocessing.set_sharing_strategy('file_system')

deviceSupportCUDA = torch.cuda.is_available()

if not deviceSupportCUDA and opt.useCUDA:
  print("Your device is not support CUDA!")
  opt.useCUDA = False

device = torch.device("cuda" if opt.useCUDA else "cpu")
print("The Test using device is {}.".format(device))
if opt.useCUDA:
  print("\tDevice: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

print("Pytorch Version {}".format(torch.__version__))

def TestImage(im_names):
    #inputpath = args.inputpath
    #inputlist = args.inputlist

    #if len(inputlist):
    #    im_names = open(inputlist, 'r').readlines()
    #elif len(inputpath) and inputpath != '/':
    #    for root, dirs, files in os.walk(inputpath):
    #        im_names = files
    #else:
    #    raise IOError('Error: must contain either --indir/--list')
        

    
    # Load input images
    data_loader = ImageLoader(im_names, batchSize=args.detbatch, format='yolo').start()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch, device=device).start()
    det_processor = DetectionProcessor(det_loader).start()
    
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)

    pose_model.to(device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    writer = DataWriter(args.save_video).start()

    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))

    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split(os.sep)[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation
            
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].to(device)
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            hm = hm.cpu()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split(os.sep)[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)
        
        if args.profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
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

def TestVideo(videofile):
    #videofile = args.video
    
    #if not len(videofile):
    #    raise IOError('Error: must contain --video')

    # Load input video
    data_loader = VideoLoader(videofile, batchSize=args.detbatch).start()
    (fourcc,fps,frameSize) = data_loader.videoinfo()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch, device=device).start()
    det_processor = DetectionProcessor(det_loader).start()
    
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)

    pose_model.to(device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_'+ntpath.basename(videofile).split('.')[0]+'.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    im_names_desc =  tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split(os.sep)[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)

            # Pose Estimation
            
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].to(device)
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)

            hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split(os.sep)[-1])
            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)
        
        if args.profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
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

def TestWecam(stream):
  
  def loop():
    n = 0
    while True:
      yield n
      n += 1
      
  data_loader = WebcamLoader(stream).start()
  (fourcc,fps,frameSize) = data_loader.videoinfo()
  
  # Load detection loader
  print('Loading YOLO model..')
  sys.stdout.flush()
  det_loader = DetectionLoader(data_loader, batchSize=args.detbatch, device=device, notWebcam=False).start()
  det_processor = DetectionProcessor(det_loader, notWebcam=False).start()
  
  # Load pose model
  pose_dataset = Mscoco()
  if args.fast_inference:
  	pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
  else:
    pose_model = InferenNet(4 * 1 + 1, pose_dataset)

  pose_model.to(device)
  pose_model.eval()

  runtime_profile = {
    'dt': [],
    'pt': [],
    'pn': []
  }

  # Data writer
  save_path = os.path.join(args.outputpath, 'AlphaPose_webcam'+opt.webcam+'.avi')
  writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

  print('Starting webcam demo, press Ctrl + C to terminate...')
  sys.stdout.flush()
  im_names_desc =  tqdm(loop())
  batchSize = args.posebatch
  for i in im_names_desc:
    try:
      start_time = getTime()
      with torch.no_grad():
        (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
        if boxes is None or boxes.nelement() == 0:
          writer.save(None, None, None, None, None, orig_img, im_name.split(os.sep)[-1])
          continue

        ckpt_time, det_time = getTime(start_time)
        runtime_profile['dt'].append(det_time)

        # Pose Estimation
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % batchSize:
          leftover = 1
        num_batches = datalen // batchSize + leftover
        hm = []
        for j in range(num_batches):
          inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].to(device)
          hm_j = pose_model(inps_j)
          hm.append(hm_j)
        hm = torch.cat(hm)

        hm = hm.cpu().data
        writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split(os.sep)[-1])

        ckpt_time, post_time = getTime(ckpt_time)
        runtime_profile['pn'].append(post_time)

      if args.profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
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

def main():
  
  if not os.path.exists(args.outputpath):
    os.makedirs(args.outputpath)
    
  # for wecam
  if opt.webcam != -1:
    stream = cv2.VideoCapture(int(opt.webcam))
    if stream.isOpened():
      print("Wecam Test is doing.")
      TestWecam(stream)
      return
    
  # for video
  isTestVideo = False
  
  if len(args.video) and os.path.isfile(args.video):
    isTestVideo = True
  
  if isTestVideo:
    print("Video Test is doing.")
    TestVideo(args.video)
    return
  
  # for images
  isTestImage = False
  
  inputpath = args.inputpath
  inputlist = args.inputlist

  if len(inputlist):
    im_names = open(inputlist, 'r').readlines()
    isTestImage = True
  elif len(inputpath) and inputpath != os.sep:
    for root, dirs, files in os.walk(inputpath):
      im_names = files
    isTestImage = True
    
  if isTestImage:
    print("Image Test is doing.")
    TestImage(im_names)
  
if __name__ == '__main__':
  main()

