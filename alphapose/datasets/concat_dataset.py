# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import bisect

import torch
import torch.utils.data as data

from alphapose.models.builder import DATASET, build_dataset


@DATASET.register_module
class ConcatDataset(data.Dataset):
    """Custom Concat dataset.
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

    def __init__(self,
                 train=True,
                 dpg=False,
                 skip_empty=True,
                 **cfg):

        self._cfg = cfg
        self._subset_cfg_list = cfg['SET_LIST']
        self._preset_cfg = cfg['PRESET']
        self._mask_id = [item['MASK_ID'] for item in self._subset_cfg_list]

        self.num_joints = self._preset_cfg['NUM_JOINTS']

        self._subsets = []
        self._subset_size = [0]
        for _subset_cfg in self._subset_cfg_list:
            subset = build_dataset(_subset_cfg, preset_cfg=self._preset_cfg, train=train)
            self._subsets.append(subset)
            self._subset_size.append(len(subset))
        self.cumulative_sizes = self.cumsum(self._subset_size)

    def __getitem__(self, idx):
        assert idx >= 0
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        dataset_idx -= 1
        sample_idx = idx - self.cumulative_sizes[dataset_idx]

        sample = self._subsets[dataset_idx][sample_idx]
        img, label, label_mask, img_id, bbox = sample

        K = label.shape[0]  # num_joints from `_subsets[dataset_idx]`
        expend_label = torch.zeros((self.num_joints, *label.shape[1:]), dtype=label.dtype)
        expend_label_mask = torch.zeros((self.num_joints, *label_mask.shape[1:]), dtype=label_mask.dtype)
        expend_label[self._mask_id[dataset_idx]:self._mask_id[dataset_idx] + K] = label
        expend_label_mask[self._mask_id[dataset_idx]:self._mask_id[dataset_idx] + K] = label_mask

        return img, expend_label, expend_label_mask, img_id, bbox

    def __len__(self):
        return self.cumulative_sizes[-1]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r
