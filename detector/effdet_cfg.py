from easydict import EasyDict as edict

cfg = edict()

cfg.NMS_THRES =  0.6  # 0.6(0.713) 0.5(0.707)
cfg.CONFIDENCE = 0.2  # 0.15       0.1
cfg.NUM_CLASSES = 80
cfg.MAX_DETECTIONS = 200  # 100
