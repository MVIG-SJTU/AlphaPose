# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu and Hao-Shu Fang
# -----------------------------------------------------

"""Halpe Full-Body(136 points) Human keypoint dataset."""
import os

import numpy as np
from tkinter import _flatten

from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy

from .custom import CustomDataset


@DATASET.register_module
class coco_wholebody(CustomDataset):
    """ Halpe Full-Body(136 points) Person dataset.

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
    joint_pairs =  [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], #17 body keypoints
        [20-3, 23-3], [21-3, 24-3], [22-3, 25-3], [26-3, 42-3], [27-3, 41-3], [28-3, 40-3], [29-3, 39-3], [30-3, 38-3], 
        [31-3, 37-3], [32-3, 36-3], [33-3, 35-3], [43-3, 52-3], [44-3, 51-3], [45-3, 50-3], [46-3, 49-3], [47-3, 48-3], 
        [62-3, 71-3], [63-3, 70-3], [64-3, 69-3], [65-3, 68-3], [66-3, 73-3], [67-3, 72-3], [57-3, 61-3], [58-3, 60-3],
        [74-3, 80-3], [75-3, 79-3], [76-3, 78-3], [87-3, 89-3], [93-3, 91-3], [86-3, 90-3], [85-3, 81-3], [84-3, 82-3],
        [94-3, 115-3], [95-3, 116-3], [96-3, 117-3], [97-3, 118-3], [98-3, 119-3], [99-3, 120-3], [100-3, 121-3],
        [101-3, 122-3], [102-3, 123-3], [103-3, 124-3], [104-3, 125-3], [105-3, 126-3], [106-3, 127-3], [107-3, 128-3],
        [108-3, 129-3], [109-3, 130-3], [110-3, 131-3], [111-3, 132-3], [112-3, 133-3], [113-3, 134-3], [114-3, 135-3]]
                

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
