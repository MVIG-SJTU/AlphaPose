import os
import os.path as osp
import time

from opt import opt
from pipeline import (DetectionProcessor, Detector, ImageCropper, ImageLoader,
                      PoseEstimator, PoseProcessor)

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'

# data_dir = '/media/data_2/COCO/2017/val2017'
# img_list = sorted(os.listdir(data_dir))
# for i in range(len(img_list)):
#     img_list[i] = os.path.join(data_dir, img_list[i])
# img_list= ['examples/demo/1.jpg','examples/demo/2.jpg','examples/demo/3.jpg']
img_list = open('examples/list-coco-minival500.txt', 'r').readlines()
img_list = [osp.join('./data/coco/images', item.strip('\n')) for item in img_list]


dataloader = ImageLoader(img_list, batch_size=opt.detbatch)
detector = Detector(dataloader)
det_processor = DetectionProcessor(detector)
img_cropper = ImageCropper(det_processor)
pose_estimator = PoseEstimator(img_cropper, batch_size=opt.posebatch)
pose_processor = PoseProcessor(pose_estimator, queue_size=5000)

dataloader.start()
detector.start()
det_processor.start()
img_cropper.start()
pose_estimator.start()
pose_processor.start()

vis_dir = 'examples/res'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

while not pose_processor.stopped:
    time.sleep(1)
