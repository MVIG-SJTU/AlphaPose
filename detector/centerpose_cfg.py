from detector.centerpose.config import cfg, update_config


update_config(cfg, './detector/centerpose/config/res_50_512x512.yaml')
cfg.defrost()
cfg.TEST.MODEL_PATH = './detector/centerpose/data/res50_cloud_99.pth'
cfg.DEBUG = 1
cfg.freeze()









