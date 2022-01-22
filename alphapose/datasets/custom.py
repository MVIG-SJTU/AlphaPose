# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com) and Haoyi Zhu
# -----------------------------------------------------

"""Custum training dataset."""
import copy
import os
import pickle as pk
from abc import abstractmethod, abstractproperty

import torch.utils.data as data
from pycocotools.coco import COCO

from alphapose.utils.presets import SimpleTransform

import cv2
import json
import numpy as np
import random

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
        self._ann_file = cfg['ANN']
        self._num_datasets = 1

        if isinstance(self._ann_file, list):
            self._num_datasets = 2
            self._root_2 = self._root[1]
            self._img_prefix_2 = self._img_prefix[1]
            self._ann_file_2 = self._ann_file[1]
            self._root = self._root[0]
            self._img_prefix = self._img_prefix[0]
            self._ann_file = self._ann_file[0]

            self._ann_file = os.path.join(self._root, self._ann_file)
            self._ann_file_2 = os.path.join(self._root_2, self._ann_file_2)
        else:
            self._ann_file = os.path.join(self._root, self._ann_file)

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
        # get image id
        if type(self._items[idx]) == dict:
            img_path = self._items[idx]['path']
            img_id = self._items[idx]['id']
        else:
            img_path = self._items[idx]
            img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # transform ground truth into training label and apply data augmentation
        img, label, label_mask, bbox = self.transformation(img, label)
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
        postfix = '_annot_keypoint.pkl' if self._num_datasets == 1 else '_plus_annot_keypoint.pkl'
        if os.path.exists(self._ann_file + postfix) and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + postfix, 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            if os.access(self._ann_file + postfix, os.W_OK):
                with open(self._ann_file + postfix, 'wb') as fid:
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