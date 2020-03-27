from __future__ import absolute_import, division, print_function

from yacs.config import CfgNode as CN

_C = CN()

_C.TASK = 'multi_pose'
_C.SAMPLE_METHOD = 'coco_hp'
_C.DATA_DIR = '/data'
_C.EXP_ID = 'default'
_C.DEBUG = 0
_C.DEBUG_THEME = 'white'
_C.TEST = False
_C.SEED = 317
_C.SAVE_RESULTS = False

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.EXPERIMENT_NAME = ''
_C.GPUS = [0, 1, 2, 3]
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.ENABLED = True
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.PRETRAINED = ''
_C.MODEL.INIT_WEIGHTS = False
_C.MODEL.NAME = 'res_50'
# 0 for no conv layer, -1 for defaults setting, 64 for resnets and 256 for dla
_C.MODEL.HEAD_CONV = 64
_C.MODEL.INTERMEDIATE_CHANNEL = 64
_C.MODEL.NUM_STACKS = 1
_C.MODEL.HEADS_NAME = 'keypoint'
_C.MODEL.HEADS_NUM = [1, 2, 34, 2, 17, 2]
_C.MODEL.DOWN_RATIO = 4
_C.MODEL.INPUT_RES = 512
_C.MODEL.OUTPUT_RES = 128
_C.MODEL.INPUT_H = 512
_C.MODEL.INPUT_W = 512
_C.MODEL.PAD = 31
_C.MODEL.NUM_CLASSES = 1
_C.MODEL.NUM_KEYPOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.SIGMA = 2
_C.MODEL.CENTER_THRESH = 0.1
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.METRIC = 'loss'
_C.LOSS.MSE_LOSS = False
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# multi pose
_C.LOSS.HP_WEIGHT = 1.
_C.LOSS.HM_WEIGHT = 1.
_C.LOSS.REG_LOSS = 'l1'
_C.LOSS.HM_HP_WEIGHT = 1.
_C.LOSS.DENSE_HP = False
_C.LOSS.HM_HP = True
_C.LOSS.REG_HP_OFFSET = True
_C.LOSS.REG_BBOX = True
_C.LOSS.WH_WEIGHT = 0.1
_C.LOSS.REG_OFFSET = True
_C.LOSS.OFF_WEIGHT = 1.


# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'coco_hp'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.TRAIN_IMAGE_DIR = 'images/train2017'
_C.DATASET.TRAIN_ANNOTATIONS = ['person_keypoints_train2017.json']
_C.DATASET.VAL_IMAGE_DIR = 'images/val2017'
_C.DATASET.VAL_ANNOTATIONS = 'person_keypoints_val2017.json'
# training data augmentation
_C.DATASET.MEAN = [0.408, 0.447, 0.470]
_C.DATASET.STD = [0.289, 0.274, 0.278]
_C.DATASET.RANDOM_CROP = True
_C.DATASET.SHIFT = 0.1
_C.DATASET.SCALE = 0.4
_C.DATASET.ROTATE = 0.
# for pose
_C.DATASET.AUG_ROT = 0.
_C.DATASET.FLIP = 0.5
_C.DATASET.NO_COLOR_AUG = False
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.SCALE_MIN = 0.5
_C.DATASET.SCALE_MAX = 1.1
_C.DATASET.IMAGE_SIZE = 512

# train
_C.TRAIN = CN()

_C.TRAIN.DISTRIBUTE = True
_C.TRAIN.LOCAL_RANK = 0
_C.TRAIN.HIDE_DATA_TIME = False
_C.TRAIN.SAVE_ALL_MODEL = False
_C.TRAIN.RESUME = False
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 120]
_C.TRAIN.EPOCHS = 140
_C.TRAIN.NUM_ITERS = -1
_C.TRAIN.LR = 1.25e-4
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.MASTER_BATCH_SIZE = -1


_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0


# 'apply and reset gradients every n batches'
_C.TRAIN.STRIDE_APPLY = 1

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.SHUFFLE = True
_C.TRAIN.VAL_INTERVALS = 5
_C.TRAIN.TRAINVAL = False

# testing
_C.TEST = CN()
# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.TASK = 'multi_pose'
_C.TEST.MODEL_PATH = ''
_C.TEST.DEMO_FILE = ''
_C.TEST.MODEL_FILE = ''
_C.TEST.TEST_SCALES = [1]
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.TOPK = 100
_C.TEST.NMS = False
_C.TEST.NMS_THRE = 0.5
_C.TEST.NOT_PREFETCH_TEST = False
_C.TEST.FIX_RES = True
_C.TEST.KEEP_RES = False

_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.VIS_THRESH = 0.3
_C.TEST.KEYPOINT_THRESH = 0.2
_C.TEST.NUM_MIN_KPT = 4
_C.TEST.THRESH_HUMAN = 0.4

_C.TEST.EVAL_ORACLE_HM = False
_C.TEST.EVAL_ORACLE_WH = False
_C.TEST.EVAL_ORACLE_OFFSET = False
_C.TEST.EVAL_ORACLE_KPS = False
_C.TEST.EVAL_ORACLE_HMHP = False
_C.TEST.EVAL_ORACLE_HP_OFFSET = False
_C.TEST.EVAL_ORACLE_DEP = False


def update_config(cfg, args_cfg):

    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
