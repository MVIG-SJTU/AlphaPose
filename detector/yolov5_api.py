# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn)
# -----------------------------------------------------

"""API of YOLOv5 detector"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.preprocess import prep_image, prep_frame

from detector.apis import BaseDetector


class YOLOV5Detector(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(YOLOV5Detector, self).__init__()

        self.detector_cfg = cfg
        self.detector_opt = opt
        self.weights = cfg.get("WEIGHT", "detector/yolov5/data/yolov5l6.pt")
        self.inp_dim = cfg.get("INP_DIM", 1280)
        self.img_size = [self.inp_dim, self.inp_dim]
        self.augment = cfg.get("AUGMENT", False)
        self.confi_thres = cfg.get("CONF_THRES", 0.05)
        self.iou_thres = cfg.get("IOU_THRES", 0.6)
        self.max_det = cfg.get("MAX_DET", 300)
        self.load_model()

    def load_model(self):
        args = self.detector_opt

        # Load model
        print("Loading YOLO model..")
        self.model = DetectMultiBackend(self.weights)
        self.stride = self.model.stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check image size

        if args:
            if len(args.gpus) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus).to(
                    args.device
                )
            else:
                self.model.to(args.device)
        else:
            self.model.cuda()
        self.model.eval()

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        if isinstance(img_source, str):
            img, orig_img, im_dim_list = prep_image(
                img_source, self.img_size, self.stride
            )
        elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
            img, orig_img, im_dim_list = prep_frame(
                img_source, self.img_size, self.stride
            )
        else:
            raise IOError("Unknown image source type: {}".format(type(img_source)))

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
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()
        with torch.no_grad():
            imgs = imgs.to(args.device) if args else imgs.cuda()
            prediction = self.model(imgs, augment=self.augment, visualize=False)
            # do nms to the detection results, only human category is left
            dets = self.dynamic_write_results(
                prediction,
                conf_thres=self.confi_thres,
                iou_thres=self.iou_thres,
                classes=0,
                max_det=self.max_det,
            )
            if isinstance(dets, int) or dets.shape[0] == 0:
                return 0
            dets = dets.cpu()

            orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.inp_dim / orig_dim_list, 1)[0].view(-1, 1)
            dets[:, [1, 3]] -= (
                self.inp_dim - scaling_factor * orig_dim_list[:, 0].view(-1, 1)
            ) / 2
            dets[:, [2, 4]] -= (
                self.inp_dim - scaling_factor * orig_dim_list[:, 1].view(-1, 1)
            ) / 2
            dets[:, 1:5] /= scaling_factor
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

            return dets

    def dynamic_write_results(
        self, prediction, conf_thres, iou_thres, classes=0, max_det=300
    ):
        prediction_bak = prediction.clone()
        dets = non_max_suppression(
            prediction.clone(),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            max_det=max_det,
        )
        if isinstance(dets, int):
            return dets

        if dets.shape[0] > 100:
            iou_thres -= 0.05
            dets = non_max_suppression(
                prediction_bak.clone(),
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                classes=classes,
                max_det=max_det,
            )

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
        # pre-process(scale, normalize, ...) the image
        img, orig_img, img_dim_list = prep_image(img_name, self.inp_dim, self.stride)
        with torch.no_grad():
            img_dim_list = torch.FloatTensor([img_dim_list]).repeat(1, 2)
            img = img.to(args.device) if args else img.cuda()
            prediction = self.model(img, augment=self.augment, visualize=False)
            # do nms to the detection results, only human category is left
            dets = self.dynamic_write_results(
                prediction,
                conf_thres=self.confi_thres,
                iou_thres=self.iou_thres,
                classes=0,
                max_det=self.max_det,
            )
            if isinstance(dets, int) or dets.shape[0] == 0:
                return None
            dets = dets.cpu()

            img_dim_list = torch.index_select(img_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.inp_dim / img_dim_list, 1)[0].view(-1, 1)
            dets[:, [1, 3]] -= (
                self.inp_dim - scaling_factor * img_dim_list[:, 0].view(-1, 1)
            ) / 2
            dets[:, [2, 4]] -= (
                self.inp_dim - scaling_factor * img_dim_list[:, 1].view(-1, 1)
            ) / 2
            dets[:, 1:5] /= scaling_factor
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, img_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, img_dim_list[i, 1])

                # write results
                det_dict = {}
                x = float(dets[i, 1])
                y = float(dets[i, 2])
                w = float(dets[i, 3] - dets[i, 1])
                h = float(dets[i, 4] - dets[i, 2])
                det_dict["category_id"] = 1
                det_dict["score"] = float(dets[i, 5])
                det_dict["bbox"] = [x, y, w, h]
                det_dict["image_id"] = int(os.path.basename(img_name).split(".")[0])
                dets_results.append(det_dict)

            return dets_results
