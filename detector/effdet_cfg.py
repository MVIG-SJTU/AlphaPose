from easydict import EasyDict as edict

cfg = edict()

cfg.NMS_THRES =  0.5  # 0.6
cfg.CONFIDENCE = 0.05  # 0.1
cfg.NUM_CLASSES = 80
cfg.MAX_DETECTIONS = 200  # 100
