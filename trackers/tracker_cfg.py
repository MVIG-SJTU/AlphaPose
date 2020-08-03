from easydict import EasyDict as edict
cfg = edict()
cfg.nid = 1000
cfg.arch = "osnet" # "osnet" or "res50-fc512"
cfg.loadmodel = "trackers/osnet_x1_0_imagenet.pth"
cfg.frame_rate =  30
cfg.track_buffer = 30 
cfg.conf_thres = 0.5
cfg.nms_thres = 0.4
cfg.iou_thres = 0.5