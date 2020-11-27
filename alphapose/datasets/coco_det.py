# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

"""MS COCO Human Detection Box dataset."""
import json
import os

import cv2
import torch
import torch.utils.data as data
from tqdm import tqdm

from alphapose.utils.presets import SimpleTransform
from detector.apis import get_detector
from alphapose.models.builder import DATASET


@DATASET.register_module
class Mscoco_det(data.Dataset):
    """ COCO human detection box dataset.

    """
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def __init__(self,
                 det_file=None,
                 opt=None,
                 **cfg):

        self._cfg = cfg
        self._opt = opt
        self._preset_cfg = cfg['PRESET']
        self._root = cfg['ROOT']
        self._img_prefix = cfg['IMG_PREFIX']
        if not det_file:
            det_file = cfg['DET_FILE']
        self._ann_file = os.path.join(self._root, cfg['ANN'])

        if os.path.exists(det_file):
            print("Detection results exist, will use it")
        else:
            print("Will create detection results to {}".format(det_file))
            self.write_coco_json(det_file)

        assert os.path.exists(det_file), "Error: no detection results found"
        with open(det_file, 'r') as fid:
            self._det_json = json.load(fid)

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']

        self._sigma = self._preset_cfg['SIGMA']

        if self._preset_cfg['TYPE'] == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False)

    def __getitem__(self, index):
        det_res = self._det_json[index]
        if not isinstance(det_res['image_id'], int):
            img_id, _ = os.path.splitext(os.path.basename(det_res['image_id']))
            img_id = int(img_id)
        else:
            img_id = det_res['image_id']
        img_path = './data/coco/val2017/%012d.jpg' % img_id

        # Load image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) #scipy.misc.imread(img_path, mode='RGB')

        imght, imgwidth = image.shape[1], image.shape[2]
        x1, y1, w, h = det_res['bbox']
        bbox = [x1, y1, x1 + w, y1 + h]
        inp, bbox = self.transformation.test_transform(image, bbox)
        return inp, torch.Tensor(bbox), torch.Tensor([det_res['bbox']]), torch.Tensor([det_res['image_id']]), torch.Tensor([det_res['score']]), torch.Tensor([imght]), torch.Tensor([imgwidth])

    def __len__(self):
        return len(self._det_json)

    def write_coco_json(self, det_file):
        from pycocotools.coco import COCO
        import pathlib

        _coco = COCO(self._ann_file)
        image_ids = sorted(_coco.getImgIds())
        det_model = get_detector(self._opt)
        dets = []
        for entry in tqdm(_coco.loadImgs(image_ids)):
            abs_path = os.path.join(
                self._root, self._img_prefix, entry['file_name'])
            det = det_model.detect_one_img(abs_path)
            if det:
                dets += det
        pathlib.Path(os.path.split(det_file)[0]).mkdir(parents=True, exist_ok=True)
        json.dump(dets, open(det_file, 'w'))

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
