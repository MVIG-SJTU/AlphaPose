# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn)
# -----------------------------------------------------

"""API of YOLOX detector"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from yolox.yolox.exp import get_exp
from yolox.utils import prep_image, prep_frame
from yolox.yolox.utils import postprocess

from detector.apis import BaseDetector


class YOLOXDetector(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(YOLOXDetector, self).__init__()

        self.detector_cfg = cfg
        self.detector_opt = opt
        self.model_name = cfg.get("MODEL_NAME", "yolox-x")
        self.model_weights = cfg.get("MODEL_WEIGHTS", "detector/yolox/data/yolox_x.pth")
        self.exp = get_exp(exp_name=self.model_name)
        self.num_classes = self.exp.num_classes
        self.conf_thres = cfg.get("CONF_THRES", 0.1)
        self.nms_thres = cfg.get("NMS_THRES", 0.6)
        self.inp_dim = cfg.get("INP_DIM", 640)
        self.img_size = [self.inp_dim, self.inp_dim]

        self.model = None

    def load_model(self):
        args = self.detector_opt

        # Load model
        print(f"Loading {self.model_name.upper().replace('_', '-')} model..")
        self.model = self.exp.get_model()
        self.model.load_state_dict(
            torch.load(self.model_weights, map_location="cpu")["model"]
        )

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
            img, orig_img, im_dim_list = prep_image(img_source, self.img_size)
        elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
            img, orig_img, im_dim_list = prep_frame(img_source, self.img_size)
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
            prediction = self.model(imgs)
            # do nms to the detection results, only human category is left
            dets = self.dynamic_write_results(
                prediction,
                num_classes=self.num_classes,
                conf_thres=self.conf_thres,
                nms_thres=self.nms_thres,
                classes=0,
            )
            if isinstance(dets, int) or dets.shape[0] == 0:
                return 0
            dets = dets.cpu()

            orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.inp_dim / orig_dim_list, 1)[0].view(-1, 1)
            dets[:, 1:5] /= scaling_factor
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

            return dets

    def dynamic_write_results(
        self, prediction, num_classes, conf_thres, nms_thres, classes=0
    ):
        prediction_bak = prediction.clone()
        dets = postprocess(
            prediction.clone(),
            num_classes=num_classes,
            conf_thre=conf_thres,
            nms_thre=nms_thres,
            classes=classes,
        )
        if isinstance(dets, int):
            return dets

        if dets.shape[0] > 100:
            nms_thres -= 0.05
            dets = postprocess(
                prediction.clone(),
                num_classes=num_classes,
                conf_thre=conf_thres,
                nms_thre=nms_thres,
                classes=classes,
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
        img, orig_img, img_dim_list = prep_image(img_name, self.img_size)
        with torch.no_grad():
            img_dim_list = torch.FloatTensor([img_dim_list]).repeat(1, 2)
            img = img.to(args.device) if args else img.cuda()
            prediction = self.model(img)
            # do nms to the detection results, only human category is left
            dets = self.dynamic_write_results(
                prediction,
                num_classes=self.num_classes,
                conf_thres=self.conf_thres,
                nms_thres=self.nms_thres,
                classes=0,
            )
            if isinstance(dets, int) or dets.shape[0] == 0:
                return None
            dets = dets.cpu()

            img_dim_list = torch.index_select(img_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.inp_dim / img_dim_list, 1)[0].view(-1, 1)
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
