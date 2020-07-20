import cv2
import numpy as np
from distutils.version import LooseVersion
import torch
from torch.autograd import Variable

from utils import bbox as bbox_utils
from utils.log import logger
from ReidModels import net_utils
from ReidModels.reid.image_part_aligned import Model


def load_reid_model():
    model = Model(n_parts=8)
    model.inp_size = (80, 160)
    ckpt = 'data/googlenet_part8_all_xavier_ckpt_56.h5'

    net_utils.load_net(ckpt, model)
    logger.info('Load ReID model from {}'.format(ckpt))

    model = model.cuda()
    model.eval()
    return model


def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
    image = image.transpose((2, 0, 1))
    return image


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = bbox_utils.clip_boxes(bboxes, image.shape)
    patches = [image[box[1]:box[3], box[0]:box[2]] for box in bboxes]
    return patches


def extract_reid_features(reid_model, image, tlbrs):
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    patches = extract_image_patches(image, tlbrs)
    patches = np.asarray([im_preprocess(cv2.resize(p, reid_model.inp_size)) for p in patches], dtype=np.float32)

    gpu = net_utils.get_device(reid_model)
    with torch.no_grad():
        _img = torch.from_numpy(patches)
        if gpu:
            _img = _img.cuda()
        features,id = reid_model(_img).detach()
    return features
