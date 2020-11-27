# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com), Haoyi Zhu
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
        if os.path.exists('/home/group3/background.json'):
            self.bgim = json.load(open('/home/group3/background.json','r'))
        else:
            self.bgim = None

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
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        if self.bgim and (source == 'frei' or source == 'partX' or source == 'OneHand' or source == 'interhand'): # hand
            img, label = self.hand_augmentation(img, label)
            
        if source == 'hand_labels_synth' or source == 'hand143_panopticdb': # hand
            if not self.skip_augmentation(0.8):
                img = self.motion_blur(img)
        
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

    def motion_blur(self, image, degree=12, angle=45):
        image = np.array(image)
     
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
     
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
     
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

    def skip_augmentation(self, p):
        x = np.random.rand()
        if x < p:
            return True
        else:
            return False

    def get_bgimg(self, box_h, box_w):
        bgimgpath = random.choice(self.bgim)
        file_name = bgimgpath['file_name']
        img_name = file_name.split('/')[-1]
        img_path = '/home/group3/coco/train2017/' + img_name
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[0], img.shape[1]

        if img_h <= box_h or img_w <= box_w:
            img = cv2.resize(img, (int(max(img_w, (np.random.rand()*1.8+1.2) * box_w)), int(max(img_h, (np.random.rand()*1.8+1.2) * box_h))))
            img_h, img_w = img.shape[0], img.shape[1]

        # crop the img
        if (img_h <= img_w) and (img_h - 4 > box_w) and (np.random.rand() > 0.25):
            crop_w = int((img_h - 2 - max((box_w + 2), (img_h / 3))) * np.random.rand() + max((box_w + 2), (img_h / 3)))
            start_p = (img_w - crop_w) * np.random.rand()
            img = img[:, int(start_p):int(start_p + crop_w + 1), :]
            assert img.shape[1] > box_w and img.shape[0] > img.shape[1], (img.shape, (box_w, box_h))

        assert img.shape[0] > box_h and img.shape[1] > box_w, (img.shape, (box_h, box_w))

        return img

    def hand_augmentation(self, img, label):
        # some images are too big (mainly in OneHand)
        if img.shape[0] > 640 or img.shape[1] > 640:
            h, w, c = img.shape
            resize_scale = 640 / h
            img = cv2.resize(img, (int(w * resize_scale), int(h * resize_scale)))
            handkp = label['joints_3d'][:,0:2,0][115:136,:]
            assert handkp.shape == (21, 2)
            handkp = handkp * resize_scale
            label['joints_3d'][:,0:2,0][115:136,:] = handkp
            label['bbox'] = list(np.array(label['bbox']) * resize_scale)
            label['height'], label['width'] = img.shape[0:2]
        
        if not self.skip_augmentation(0.8):
            img = self.motion_blur(img)

        label['bbox'] = list(label['bbox'])
        # print(img_path, 'hand augmentation')
        if not self.skip_augmentation(0.3):
            handkp = label['joints_3d'][:,0:2,0][115:136,:]
            assert handkp.shape == (21, 2)
            
            # resize the hand img (random scale between 40% and 100%)
            resize_scale = 0.6 * np.random.rand() + 0.4
            handkp = handkp * resize_scale
            img = cv2.resize(img, dsize=None, fx=resize_scale, fy=resize_scale)
            label['height'], label['width'] = img.shape[0:2]
            label['bbox'] = list(np.array(label['bbox']) * resize_scale)

            h, w = img.shape[0:2]
            hand_xmin, hand_xmax, hand_ymin, hand_ymax = int(round(min(handkp[:,0]))), int(round(max(handkp[:,0]))),int(round(min(handkp[:,1]))), int(round(max(handkp[:,1])))
            boxw_time, boxh_time = float(np.random.rand()*2+4), float(np.random.rand()*4+5)
            box_w, box_h = max(int((hand_xmax - hand_xmin)*boxw_time), w+1), max(int((hand_ymax - hand_ymin)*boxh_time),h+1)

            background= self.get_bgimg(box_h, box_w)

            bh, bw, bc = background.shape
            # print(bw - box_w, bh - box_h)
            x, y = int(np.random.randint(0,int(bw - box_w),size=1)), int(np.random.randint(0,int(bh - box_h), size=1))
            hd = copy.deepcopy(img)
            new_image = copy.deepcopy(background)
            ralative_x, ralative_y = int(np.random.randint(0,int(box_w-w),size=1)), int(np.random.randint(0,int(box_h-h), size=1))
            new_loc_x, new_loc_y = x + ralative_x, y + ralative_y
            assert  (new_loc_x+w < x+box_w) and (new_loc_y+h < y+box_h)

            handkp[(handkp[:, 0] + handkp[:, 1]) > 0] += [new_loc_x, new_loc_y]

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

            max_length = max(new_image.shape[0], new_image.shape[1])
            if max_length > 1200:
                scale = 640 / max_length
                new_image = cv2.resize(new_image, (int(round(new_image.shape[1] * scale)), int(round(new_image.shape[0] * scale))))
                handkp = handkp * scale
            label['joints_3d'][:,0:2,0][115:136,:] = handkp
            img = new_image
            label['height'], label['width'] = img.shape[0:2]

        label['bbox'] = tuple(label['bbox'])
        assert label['height'] == img.shape[0] and label['width'] == img.shape[1], (img.shape, (label['height'], label['width']), flag)

        return img, label

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
