from easydict import EasyDict as edict
import os

current_directory = os.getcwd()

config_relative_path = os.path.join(
    current_directory, "AlphaPose/detector/yolo/cfg/yolov3-spp.cfg")
weight_relative_path = os.path.join(
    current_directory, "AlphaPose/detector/yolo/data/yolov3-spp.weights")


cfg = edict()
cfg.CONFIG = config_relative_path
cfg.WEIGHTS = weight_relative_path
cfg.INP_DIM = 608
cfg.NMS_THRES = 0.6
cfg.CONFIDENCE = 0.02
cfg.NUM_CLASSES = 80
