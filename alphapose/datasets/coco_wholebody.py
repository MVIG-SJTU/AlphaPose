# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu and Hao-Shu Fang
# -----------------------------------------------------

"""Coco WholeBody (133 points) Human keypoint dataset."""
import os

import numpy as np
from tkinter import _flatten

from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy

from .custom import CustomDataset


@DATASET.register_module
class coco_wholebody(CustomDataset):
    """ Coco WholeBody (133 points) Person dataset.

    Parameters
    ----------
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.
    dpg: bool, default is False
        If true, will activate `dpg` for data augmentation.
    """
    CLASSES = ['person']
    EVAL_JOINTS = list(range(133))
    num_joints = 133
    CustomDataset.lower_body_ids = (11, 12, 13, 14, 15, 16, 17, 21-3, 22-3, 23-3, 24-3, 25-3)
    """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
    joint_pairs =  [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], 
                    [17, 20], [18, 21], [19, 22], [23, 39], [24, 38], [25, 37], [26, 36], 
                    [27, 35], [28, 34], [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], 
                    [43, 46], [44, 45], [59, 68], [60, 67], [61, 66], [62, 65], [63, 70], 
                    [64, 69], [54, 58], [55, 57], [71, 77], [72, 76], [73, 75], [84, 86], 
                    [90, 88], [83, 87], [82, 78], [81, 79], [91, 112], [92, 113], [93, 114], 
                    [94, 115], [95, 116], [96, 117], [97, 118], [98, 119], [99, 120], 
                    [100, 121], [101, 122], [102, 123], [103, 124], [104, 125], [105, 126], 
                    [106, 127], [107, 128], [108, 129], [109, 130], [110, 131], [111, 132]]
                

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        _coco = self._lazy_load_ann_file()

        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with COCO. "

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}

        # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join('/DATA1/Benchmark/coco', dirname, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_keypoints(_coco, entry)
            if not label:
                continue
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels

    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            #obj['keypoints'].extend([0,0,0, 0,0,0, 0,0,0])
            if 'foot_kpts' in obj and 'face_kpts' in obj and 'lefthand_kpts' in obj and 'righthand_kpts' in obj:
                obj['keypoints'].extend(obj['foot_kpts'])
                obj['keypoints'].extend(obj['face_kpts'])
                obj['keypoints'].extend(obj['lefthand_kpts'])
                obj['keypoints'].extend(obj['righthand_kpts'])
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            #if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
            if (xmax-xmin)*(ymax-ymin) <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                if obj['keypoints'][i * 3 + 2] >= 0.35:
                    visible = 1
                else:
                    visible = 0
                #visible = min(1, visible)
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
