# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Chao Xu (xuchao.19962007@sjtu.edu.cn)
# -----------------------------------------------------

"""API of detector"""
from abc import ABC, abstractmethod


def get_detector(opt=None):
    if opt.detector == 'yolo':
        from detector.yolo_api import YOLODetector
        from detector.yolo_cfg import cfg
        return YOLODetector(cfg, opt)
    elif opt.detector == 'centerpose':
        from detector.centerpose_api import CenterposeDetector 
        from detector.centerpose_cfg import cfg
        return CenterposeDetector(cfg)
    elif opt.detector == 'efficientdet':
        from detector.efficientdet_api import EffDetector
        return EffDetector(7)
    elif opt.detector == 'tracker':
        from detector.tracker_api import Tracker
        from detector.tracker_cfg import cfg
        return Tracker(cfg, opt)
    else:
        raise NotImplementedError


class BaseDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_preprocess(self, img_name):
        pass

    @abstractmethod
    def images_detection(self, imgs, orig_dim_list):
        pass

    @abstractmethod
    def detect_one_img(self, img_name):
        pass
