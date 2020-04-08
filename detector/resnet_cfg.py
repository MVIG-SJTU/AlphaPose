from detector.centorpose.config import cfg, update_config


update_config(cfg, './detector/centorpose/config/res_50_512x512.yaml')
cfg.defrost()
cfg.TEST.MODEL_PATH = './detector/centorpose/data/res50_cloud_99.pth'
cfg.DEBUG = 1
cfg.freeze()









