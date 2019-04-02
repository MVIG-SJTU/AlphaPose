import gluoncv as gcv
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'

import time
import numpy as np
import matplotlib.pyplot as plt
from pipeline import VideoLoader, Detector, DetectionProcessor, ImageCropper, PoseEstimator, PoseProcessor

dataloader = VideoLoader('examples/video/LondonStreet.mp4', batch_size=1)
detector = Detector(dataloader)
det_processor = DetectionProcessor(detector)
img_cropper = ImageCropper(det_processor)
pose_estimator = PoseEstimator(img_cropper, batch_size=80)
pose_processor = PoseProcessor(pose_estimator, queue_size=1000)

dataloader.start()
detector.start()
det_processor.start()
img_cropper.start()
pose_estimator.start()
pose_processor.start()

vis_dir = 'examples/res'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

cnt = 0
while not pose_processor.stopped:
    try:
        img, final_result, boxes, box_scores, img_name = pose_processor.next(timeout=1)
    except:
        continue