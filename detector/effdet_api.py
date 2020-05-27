# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoshu Fang (fhaoshu@gmail.com)
# -----------------------------------------------------

"""API of efficientdet detector"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from abc import ABC, abstractmethod
import platform

import torch
import numpy as np

from efficientdet.utils import unique, prep_image, prep_frame, bbox_iou
from efficientdet.effdet import EfficientDet, get_efficientdet_config, DetBenchEval, load_checkpoint

from detector.apis import BaseDetector

try:
    from apex import amp
    has_amp = True
except ImportError:
    has_amp = False

#only windows visual studio 2013 ~2017 support compile c/cuda extensions
#If you force to compile extension on Windows and ensure appropriate visual studio
#is intalled, you can try to use these ext_modules.
if platform.system() != 'Windows':
    from detector.nms import nms_wrapper


class EffDetDetector(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(EffDetDetector, self).__init__()

        self.detector_cfg = cfg
        self.detector_opt = opt
        self.model_cfg = get_efficientdet_config(opt.detector)
        self.model_weights = 'detector/efficientdet/weights/'+opt.detector+'.pth'
        #Input image dimension, uses model default if empty
        self.inp_dim = cfg.get('INP_DIM', None) if cfg.get('INP_DIM', None) is not None else self.model_cfg.image_size
        self.nms_thres = cfg.get('NMS_THRES', 0.6)
        self.confidence = cfg.get('CONFIDENCE', 0.05)
        self.num_classes = cfg.get('NUM_CLASSES', 80)
        self.max_dets = cfg.get('MAX_DETECTIONS', 100)
        self.model = None


    def load_model(self):
        args = self.detector_opt

        net = EfficientDet(self.model_cfg)
        load_checkpoint(net, self.model_weights)
        self.model = DetBenchEval(net, self.model_cfg, nms_thres=self.nms_thres, max_dets=self.max_dets)

        if args:
            if len(args.gpus) > 1:
                if has_amp:
                    print('Using AMP mixed precision.')
                    self.model = amp.initialize(self.model, opt_level='O1')
                else:
                    print('AMP not installed, running network in FP32.')

                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus).to(args.device)
            else:
                self.model.to(args.device)
        else:
            if has_amp:
                print('Using AMP mixed precision.')
                self.model = amp.initialize(self.model, opt_level='O1')
            else:
                print('AMP not installed, running network in FP32.')
            self.model.cuda()

        net.eval()

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        if isinstance(img_source, str):
            img, orig_img, im_dim_list = prep_image(img_source, self.inp_dim)
        elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
            img, orig_img, im_dim_list = prep_frame(img_source, self.inp_dim)
        else:
            raise IOError('Unknown image source type: {}'.format(type(img_source)))

        return img

    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into object detection network and 
        collect bbox w.r.t original image size
        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size
        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results
        """
        args = self.detector_opt
        if not self.model:
            self.load_model()
        with torch.no_grad():
            imgs = imgs.to(args.device) if args else imgs.cuda()
            scaling_factors = torch.FloatTensor([1./min(self.inp_dim / orig_dim[0], self.inp_dim / orig_dim[1]) for orig_dim in orig_dim_list]).view(-1, 1)
            scaling_factors = scaling_factors.to(args.device) if args else scaling_factors.cuda()
            prediction = self.model(imgs, scaling_factors) 
            #change the pred format to alphapose (nms has already been done in effdeteval model)
            prediction = prediction.cpu()
            write = False
            for index, sample in enumerate(prediction):
                for det in sample:
                    score = float(det[4])
                    if score < .001:  # stop when below this threshold, scores in descending order
                        break
                    if int(det[5]) != 1 or score < self.confidence:
                        continue
                    det_new = prediction.new(1,8)
                    det_new[0,0] = index    #index of img
                    det_new[0,1:3] = det[0:2]  # bbox x1,y1
                    det_new[0,3:5] = det[0:2] + det[2:4] # bbox x2,y2
                    det_new[0,6:7] = det[4]  # cls conf
                    det_new[0,7] = det[5]   # cls idx
                    if not write:
                        dets = det_new
                        write = True
                    else:
                        dets = torch.cat((dets, det_new))    
            if not write:
                return 0

            orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

            return dets

    def detect_one_img(self, img_name):
        """
        Detect bboxs in one image
        Input: 'str', full path of image
        Output: '[{"category_id":1,"score":float,"bbox":[x,y,w,h],"image_id":str},...]',
        The output results are similar with coco results type, except that image_id uses full path str
        instead of coco %012d id for generalization. 
        """
        args = self.detector_opt
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        dets_results = []
        #pre-process(scale, normalize, ...) the image
        img, orig_img, img_dim_list = prep_image(img_name, self.inp_dim)
        with torch.no_grad():
            img_dim_list = torch.FloatTensor([img_dim_list]).repeat(1, 2)
            img = img.to(args.device) if args else img.cuda()
            scaling_factor = torch.FloatTensor([1/min(self.inp_dim / orig_dim[0], self.inp_dim / orig_dim[1]) for orig_dim in img_dim_list]).view(-1, 1)
            scaling_factor = scaling_factor.to(args.device) if args else scaling_factor.cuda()
            prediction = self.model(img, scaling_factor) 
            #change the pred format to alphapose (nms has already been done in effdeteval model)
            prediction = prediction.cpu()
            write = False
            for index, sample in enumerate(prediction):
                for det in sample:
                    score = float(det[4])
                    if score < .001:  # stop when below this threshold, scores in descending order
                        break
                    if int(det[5]) != 1 or score < self.confidence:
                        continue
                    det_new = prediction.new(1,8)
                    det_new[0,0] = index    #index of img
                    det_new[0,1:3] = det[0:2]  # bbox x1,y1
                    det_new[0,3:5] = det[0:2] + det[2:4] # bbox x2,y2
                    det_new[0,6:7] = det[4]  # cls conf
                    det_new[0,7] = det[5]   # cls idx
                    if not write:
                        dets = det_new
                        write = True
                    else:
                        dets = torch.cat((dets, det_new))          
            if not write:
                return None

            img_dim_list = torch.index_select(img_dim_list, 0, dets[:, 0].long())
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, img_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, img_dim_list[i, 1])

                #write results
                det_dict = {}
                x = float(dets[i, 1])
                y = float(dets[i, 2])
                w = float(dets[i, 3] - dets[i, 1])
                h = float(dets[i, 4] - dets[i, 2])
                det_dict["category_id"] = 1
                det_dict["score"] = float(dets[i, 5])
                det_dict["bbox"] = [x, y, w, h]
                det_dict["image_id"] = int(os.path.basename(img_name).split('.')[0])
                dets_results.append(det_dict)

            return dets_results


    def check_detector(self, img_name):
        """
        Detect bboxs in one image
        Input: 'str', full path of image
        Output: '[{"category_id":1,"score":float,"bbox":[x,y,w,h],"image_id":str},...]',
        The output results are similar with coco results type, except that image_id uses full path str
        instead of coco %012d id for generalization. 
        """
        args = self.detector_opt
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        dets_results = []
        #pre-process(scale, normalize, ...) the image
        img, orig_img, img_dim_list = prep_image(img_name, self.inp_dim)
        with torch.no_grad():
            img_dim_list = torch.FloatTensor([img_dim_list]).repeat(1, 2)
            img = img.to(args.device) if args else img.cuda()
            scaling_factor = torch.FloatTensor([1/min(self.inp_dim / orig_dim[0], self.inp_dim / orig_dim[1]) for orig_dim in img_dim_list]).view(-1, 1)
            scaling_factor = scaling_factor.to(args.device) if args else scaling_factor.cuda()
            output = self.model(img, scaling_factor) 

            output = output.cpu()
            for index, sample in enumerate(output):
                image_id = int(os.path.basename(img_name).split('.')[0])
                for det in sample:
                    score = float(det[4])
                    if score < .001:  # stop when below this threshold, scores in descending order
                        break
                    #### uncomment it for only human detection
                    # if int(det[5]) != 1 or score < self.confidence:
                    #     continue
                    coco_det = dict(
                        image_id=image_id,
                        bbox=det[0:4].tolist(),
                        score=score,
                        category_id=int(det[5]))
                    dets_results.append(coco_det)

        return dets_results

if __name__ == "__main__":
#run with python detector/effdet_api.py /DATA1/Benchmark/coco/ efficientdet_d0
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from easydict import EasyDict as edict
        from apis import get_detector
        from tqdm import tqdm
        import json

        opt = edict()
        _coco = COCO(sys.argv[1]+'/annotations/instances_val2017.json')
        # _coco = COCO(sys.argv[1]+'/annotations/person_keypoints_val2017.json')
        opt.detector = sys.argv[2]
        opt.gpus = [0] if torch.cuda.device_count() >= 1 else [-1]
        opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")
        image_ids = sorted(_coco.getImgIds())
        det_model = get_detector(opt)
        dets = []
        for entry in tqdm(_coco.loadImgs(image_ids)):
            abs_path = os.path.join(
                sys.argv[1], 'val2017', entry['file_name'])
            det = det_model.check_detector(abs_path)
            if det:
                dets += det
        result_file = 'results.json'
        json.dump(dets, open(result_file, 'w'))

        coco_results = _coco.loadRes(result_file)
        coco_eval = COCOeval(_coco, coco_results, 'bbox')
        coco_eval.params.imgIds = image_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
