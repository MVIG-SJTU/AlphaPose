from easydict import EasyDict as edict
cfg = edict()
cfg.nid = 1000
cfg.loadmodel = "trackers/resnet50_fc512.pth"
cfg.frame_rate =  30
cfg.track_buffer = 30 
cfg.conf_thres = 0.5
cfg.nms_thres = 0.4
cfg.iou_thres = 0.5