from easydict import EasyDict as edict

cfg = edict()
cfg.WEIGHT = "detector/yolov5/data/yolov5l6.pt"
cfg.INP_DIM = 1280
cfg.AUGMENT = False
cfg.CONF_THRES = 0.1
cfg.IOU_THRES = 0.6
cfg.MAX_DET = 300
