# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu and Hao-Shu Fang
# -----------------------------------------------------

"""Single Hand (21 keypoints) dataset."""
import os

import numpy as np
from tkinter import _flatten

from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy

from .custom import CustomDataset


@DATASET.register_module
class SingleHand(CustomDataset):
    """ Single Hand (21 keypoints) Person dataset.

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
    EVAL_JOINTS = list(range(21))
    num_joints = 21
    CustomDataset.lower_body_ids = ()
    """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
    joint_pairs = []       

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
            abs_path = os.path.join(self._root, self._img_prefix, entry['file_name'])

            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_keypoints(_coco, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append({'path': abs_path, 'id': entry['id']})
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
            obj['keypoints'] = obj['keypoints'][-42*3:]
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            if 'bbox' not in obj:
                obj['bbox'] = [1, 1, width-1, height-1]
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)

            # require non-zero box area
            if (xmax-xmin)*(ymax-ymin) <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue

            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints * 2, 3, 2), dtype=np.float32)
            for i in range(self.num_joints * 2):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                if obj['keypoints'][i * 3 + 2] >= 0.35 and obj['keypoints'][i * 3 + 0] > 0 and obj['keypoints'][i * 3 + 1] > 0:
                    visible = 1
                else:
                    visible = 0
                joints_3d[i, :2, 1] = visible

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            # left hand
            if np.sum(joints_3d[:21, 0, 1]) >= 10:
                xmin = np.min(joints_3d[:21, 0, 0][joints_3d[:21, 0, 0] > 0])
                ymin = np.min(joints_3d[:21, 1, 0][joints_3d[:21, 1, 0] > 0])
                xmax = np.max(joints_3d[:21, 0, 0][joints_3d[:21, 0, 0] > 0])
                ymax = np.max(joints_3d[:21, 1, 0][joints_3d[:21, 1, 0] > 0])
                w = xmax - xmin
                h = ymax - ymin
                xmin = max(xmin - np.random.rand() * w / 2, 1)
                xmax = min(xmax + np.random.rand() * w / 2, width)
                ymin = max(ymin - np.random.rand() * h / 2, 1)
                ymax = min(ymax + np.random.rand() * h / 2, height)
                obj['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]

                if self._check_centers and self._train:
                    bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                    kp_center, num_vis = self._get_keypoints_center_count(joints_3d[:21, :, :])
                    ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                    if (num_vis / 80.0 + 47 / 80.0) > ks:
                        continue

                valid_objs.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'width': width,
                    'height': height,
                    'joints_3d': joints_3d[:21, :, :]
                })

            # right hand
            if np.sum(joints_3d[21:, 0, 1]) >= 10:
                xmin = np.min(joints_3d[21:, 0, 0][joints_3d[21:, 0, 0] > 0])
                ymin = np.min(joints_3d[21:, 1, 0][joints_3d[21:, 1, 0] > 0])
                xmax = np.max(joints_3d[21:, 0, 0][joints_3d[21:, 0, 0] > 0])
                ymax = np.max(joints_3d[21:, 1, 0][joints_3d[21:, 1, 0] > 0])
                w = xmax - xmin
                h = ymax - ymin
                xmin = max(xmin - np.random.rand() * w / 2, 1)
                xmax = min(xmax + np.random.rand() * w / 2, width)
                ymin = max(ymin - np.random.rand() * h / 2, 1)
                ymax = min(ymax + np.random.rand() * h / 2, height)
                obj['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]

                if self._check_centers and self._train:
                    bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                    kp_center, num_vis = self._get_keypoints_center_count(joints_3d[21:, :, :])
                    ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                    if (num_vis / 80.0 + 47 / 80.0) > ks:
                        continue

                valid_objs.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'width': width,
                    'height': height,
                    'joints_3d': joints_3d[21:, :, :]
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
