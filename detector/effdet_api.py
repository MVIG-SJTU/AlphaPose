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
        self.model = None


    def load_model(self):
        args = self.detector_opt

        net = EfficientDet(self.model_cfg)
        load_checkpoint(net, self.model_weights)
        self.model = DetBenchEval(net, self.model_cfg)

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
            #do nms to the detection results, only human category is left
            dets = self.dynamic_get_results(prediction, self.confidence, 
                                              self.num_classes, nms=True, 
                                              nms_conf=self.nms_thres)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return 0
            dets = dets.cpu()

            orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

            return dets

    def dynamic_get_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        prediction_bak = prediction.clone()
        dets = self.get_results(prediction.clone(), confidence, num_classes, nms, nms_conf)
        if isinstance(dets, int):
            return dets

        if dets.shape[0] > 100:
            nms_conf -= 0.05
            dets = self.get_results(prediction_bak.clone(), confidence, num_classes, nms, nms_conf)

        return dets

    def get_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        args = self.detector_opt
        #prediction: (batchsize, num of objects, (xc,yc,w,h,box confidence, 80 class scores))
        conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)
        prediction = prediction * conf_mask

        try:
            ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
        except:
            return 0

        #the 3rd channel of prediction: (xmin,ymin,w,h)->(x1,y1,x2,y2)
        box_a = prediction.new(prediction.shape)
        box_a[:,:,0] = prediction[:,:,0]
        box_a[:,:,1] = prediction[:,:,1]
        box_a[:,:,2] = prediction[:,:,0] + prediction[:,:,2]
        box_a[:,:,3] = prediction[:,:,1] + prediction[:,:,3]
        prediction[:,:,:4] = box_a[:,:,:4]

        batch_size = prediction.size(0)

        output = prediction.new(1, prediction.size(2) + 1)
        write = False
        num = 0
        for ind in range(batch_size):
            #select the image from the batch
            image_pred = prediction[ind]

            #Get the class having maximum score, and the index of that class
            #Get rid of num_classes softmax scores 
            #Add the class index and the class score of class having maximum score
            max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:,:5], max_conf, max_conf_score)
            #image_pred:(n,(x1,y1,x2,y2,c,s,idx of cls))
            image_pred = torch.cat(seq, 1)

            #Get rid of the zero entries
            non_zero_ind =  (torch.nonzero(image_pred[:,4]))

            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)

            #Get the various classes detected in the image
            try:
                img_classes = unique(image_pred_[:,-1])
            except:
                continue

            #WE will do NMS classwise
            #print(img_classes)
            for cls in img_classes:
                if cls != 0:
                    continue
                #get the detections with one particular class
                cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()

                image_pred_class = image_pred_[class_mask_ind].view(-1,7)

                #sort the detections such that the entry with the maximum objectness
                #confidence is at the top
                conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)

                #if nms has to be done
                if nms:
                    if platform.system() != 'Windows':
                        #We use faster rcnn implementation of nms (soft nms is optional)
                        nms_op = getattr(nms_wrapper, 'nms')
                        #nms_op input:(n,(x1,y1,x2,y2,c))
                        #nms_op output: input[inds,:], inds
                        _, inds = nms_op(image_pred_class[:,:5], nms_conf)

                        image_pred_class = image_pred_class[inds]
                    else:
                        # Perform non-maximum suppression
                        max_detections = []
                        while image_pred_class.size(0):
                            # Get detection with highest confidence and save as max detection
                            max_detections.append(image_pred_class[0].unsqueeze(0))
                            # Stop if we're at the last detection
                            if len(image_pred_class) == 1:
                                break
                            # Get the IOUs for all boxes with lower confidence
                            ious = bbox_iou(max_detections[-1], image_pred_class[1:], args)
                            # Remove detections with IoU >= NMS threshold
                            image_pred_class = image_pred_class[1:][ious < nms_conf]

                        image_pred_class = torch.cat(max_detections).data

                #Concatenate the batch_id of the image to the detection
                #this helps us identify which image does the detection correspond to 
                #We use a linear straucture to hold ALL the detections from the batch
                #the batch_dim is flattened
                #batch is identified by extra batch column

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq,1)
                    write = True
                else:
                    out = torch.cat(seq,1)
                    output = torch.cat((output,out))
                num += 1
    
        if not num:
            return 0
        #output:(n,(batch_ind,x1,y1,x2,y2,c,s,idx of cls))
        return output

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
            #do nms to the detection results, only human category is left
            dets = self.dynamic_get_results(prediction, self.confidence,
                                              self.num_classes, nms=True,
                                              nms_conf=self.nms_thres)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return None
            dets = dets.cpu()
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
        json.dump(dets, open('results.json', 'w'))

        coco_results = _coco.loadRes('results.json')
        coco_eval = COCOeval(_coco, coco_results, 'bbox')
        coco_eval.params.imgIds = image_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()