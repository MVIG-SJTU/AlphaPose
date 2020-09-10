# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

"""Custum training dataset."""
import copy
import os
import pickle as pk
from abc import abstractmethod, abstractproperty

import scipy.misc
import torch.utils.data as data
from pycocotools.coco import COCO

from alphapose.utils.presets import SimpleTransform


import json
import numpy as np
import random

def skip_augmentation():
    x = np.random.rand()
    if x < 0.3:
        return True
    else:
        return False

def get_bgimg(bgim, box_h, box_w):
    _bgim = [x for x in bgim if ((x['width'] > box_w) and (x['height'] > box_h))]
    if len(_bgim) == 0:
        bgimgpath = random.choice(bgim)['file_name']
        img = scipy.misc.imread(bgimgpath, mode='RGB')
        oversize = True
    else:
        bgimgpath = random.choice(_bgim)
        img = scipy.misc.imread(bgimgpath['file_name'], mode='RGB')
        oversize = False
    return img, oversize

class CustomDataset(data.Dataset):
    """Custom dataset.
    Annotation file must be in `coco` format.

    Parameters
    ----------
    train: bool, default is True
        If true, will set as training mode.
    dpg: bool, default is False
        If true, will activate `dpg` for data augmentation.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    cfg: dict, dataset configuration.
    """

    CLASSES = None

    def __init__(self,
                 train=True,
                 dpg=False,
                 skip_empty=True,
                 lazy_import=False,
                 **cfg):

        self._cfg = cfg
        self._preset_cfg = cfg['PRESET']
        self._root = cfg['ROOT']
        self._img_prefix = cfg['IMG_PREFIX']
        self._ann_file = os.path.join(self._root, cfg['ANN'])

        self._lazy_import = lazy_import
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        if 'AUG' in cfg.keys():
            self._scale_factor = cfg['AUG']['SCALE_FACTOR']
            self._rot = cfg['AUG']['ROT_FACTOR']
            self.num_joints_half_body = cfg['AUG']['NUM_JOINTS_HALF_BODY']
            self.prob_half_body = cfg['AUG']['PROB_HALF_BODY']
        else:
            self._scale_factor = 0
            self._rot = 0
            self.num_joints_half_body = -1
            self.prob_half_body = -1

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']

        self._sigma = self._preset_cfg['SIGMA']

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self._loss_type = self._preset_cfg.get('LOSS_TYPE', 'MSELoss')

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        if self._preset_cfg['TYPE'] == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=self._scale_factor,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type)
        else:
            raise NotImplementedError

        self._items, self._labels = self._lazy_load_json()

    def __getitem__(self, idx):
        source = None
        # get image id
        if type(self._items[idx]) == dict:
            img_path = self._items[idx]['path']
            img_id = self._items[idx]['id']
            source = self._items[idx]['source']
        else:
            img_path = self._items[idx]
            img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = scipy.misc.imread(img_path, mode='RGB')

        if source == 'frei': # hand
            label['bbox'] = list(label['bbox'])
            # print(img_path, 'hand augmentation')
            if not skip_augmentation():
                bgim = json.load(open('/home/group3/background.json','r'))
                handkp = label['joints_3d'][:,0:2,0][115:136,:]
                assert handkp.shape == (21, 2)

                h, w = img.shape[0:2]
                hand_xmin, hand_xmax, hand_ymin, hand_ymax = int(round(min(handkp[:,0]))), int(round(max(handkp[:,0]))),int(round(min(handkp[:,1]))), int(round(max(handkp[:,1])))
                boxw_time, boxh_time = float(np.random.random_sample(1)*2+4), float(np.random.random_sample(1)*4+5)
                box_w, box_h = max(int((hand_xmax - hand_xmin)*boxw_time), w+1), max(int((hand_ymax - hand_ymin)*boxh_time),h+1)
                
                background, oversize = get_bgimg(bgim, box_h, box_w)
                if oversize:
                    background, _ = get_bgimg(bgim, h, w)
                    box_h, box_w = background.shape[0:2]
                    x, y = int(np.random.randint(0,int(box_w - w),size=1)), int(np.random.randint(0,int(box_h - h), size=1))
                    hd = copy.deepcopy(img)
                    new_image = copy.deepcopy(background)
                    new_image[y:y+h, x:x+w, :] = hd
                    handkp[:,0] = handkp[:,0] + x
                    handkp[:,1] = handkp[:,1] + y
                    label['bbox'][0] = label['bbox'][0] + x
                    label['bbox'][1] = label['bbox'][1] + y
                    label['bbox'][2] = label['bbox'][2] + x
                    label['bbox'][3] = label['bbox'][3] + y
                else:
                    bh, bw, bc = background.shape

                    x, y = int(np.random.randint(0,int(bw - box_w),size=1)), int(np.random.randint(0,int(bh - box_h), size=1))
                    hd = copy.deepcopy(img)
                    new_image = copy.deepcopy(background)
                    ralative_x, ralative_y = int(np.random.randint(0,int(box_w-w),size=1)), int(np.random.randint(0,int(box_h-h), size=1))
                    new_loc_x, new_loc_y = x + ralative_x, y + ralative_y
                    assert  new_loc_x+w < x+box_w
                    assert  new_loc_y+h < y+box_h

                    handkp[:,0] = handkp[:,0] + new_loc_x
                    handkp[:, 1] = handkp[:,1] + new_loc_y

                    if new_loc_x < 0:
                        hd = hd[:,-new_loc_x:,:]
                        new_loc_x = 0
                    if new_loc_y < 0:
                        hd = hd[-new_loc_y:,:,:]
                        new_loc_y = 0
                    if new_loc_x+hd.shape[1]>new_image.shape[1]:
                        hd = hd[:, :new_image.shape[1]-new_loc_x, :]
                    if new_loc_y+hd.shape[0]>new_image.shape[0]:
                        hd = hd[:new_image.shape[0]-new_loc_y, :, :]

                    new_image[new_loc_y:new_loc_y+h,new_loc_x:new_loc_x+w,:] = hd
                    label['bbox'][0] = label['bbox'][0] + new_loc_x
                    label['bbox'][1] = label['bbox'][1] + new_loc_y
                    label['bbox'][2] = label['bbox'][2] + new_loc_x
                    label['bbox'][3] = label['bbox'][3] + new_loc_y
                label['joints_3d'][:,0:2,0][115:136,:] = handkp
                img = new_image
                label['height'], label['width'] = img.shape[0:2]
            label['bbox'] = tuple(label['bbox'])

        # transform ground truth into training label and apply data augmentation
        img, label, label_mask, bbox = self.transformation(img, label, source)
        return img, label, label_mask, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_ann_file(self):
        if os.path.exists(self._ann_file + '.pkl') and self._lazy_import:
            print('Lazy load json...')
            with open(self._ann_file + '.pkl', 'rb') as fid:
                return pk.load(fid)
        else:
            _database = COCO(self._ann_file)
            if os.access(self._ann_file + '.pkl', os.W_OK):
                with open(self._ann_file + '.pkl', 'wb') as fid:
                    pk.dump(_database, fid, pk.HIGHEST_PROTOCOL)
            return _database

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            if os.access(self._ann_file + '_annot_keypoint.pkl', os.W_OK):
                with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)

        return items, labels

    @abstractmethod
    def _load_jsons(self):
        pass

    @abstractproperty
    def CLASSES(self):
        return None

    @abstractproperty
    def num_joints(self):
        return None

    @abstractproperty
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return None
