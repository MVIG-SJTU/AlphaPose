from easydict import EasyDict as edict

cfg = edict()
cfg.MODEL_NAME = "yolox-x"
cfg.MODEL_WEIGHTS = "detector/yolox/data/yolox_x.pth"
cfg.INP_DIM = 640
cfg.CONF_THRES = 0.1
cfg.NMS_THRES = 0.6
