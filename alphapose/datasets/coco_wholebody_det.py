# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu
# -----------------------------------------------------

"""Haple_136 Human Detection Box dataset."""
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
class coco_wholebody_det(data.Dataset):
    """ Halpe_136 human detection box dataset.

    """
    EVAL_JOINTS = list(range(133))

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
        img_path = '/DATA1/Benchmark/coco/val2017/%012d.jpg' % img_id

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
                '/DATA1/Benchmark/coco', self._img_prefix, entry['file_name'])
            det = det_model.detect_one_img(abs_path)
            if det:
                dets += det
        pathlib.Path(os.path.split(det_file)[0]).mkdir(parents=True, exist_ok=True)
        json.dump(dets, open(det_file, 'w'))

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], #17 body keypoints
        [20-3, 23-3], [21-3, 24-3], [22-3, 25-3], [26-3, 42-3], [27-3, 41-3], [28-3, 40-3], [29-3, 39-3], [30-3, 38-3], 
        [31-3, 37-3], [32-3, 36-3], [33-3, 35-3], [43-3, 52-3], [44-3, 51-3], [45-3, 50-3], [46-3, 49-3], [47-3, 48-3], 
        [62-3, 71-3], [63-3, 70-3], [64-3, 69-3], [65-3, 68-3], [66-3, 73-3], [67-3, 72-3], [57-3, 61-3], [58-3, 60-3],
        [74-3, 80-3], [75-3, 79-3], [76-3, 78-3], [87-3, 89-3], [93-3, 91-3], [86-3, 90-3], [85-3, 81-3], [84-3, 82-3],
        [94-3, 115-3], [95-3, 116-3], [96-3, 117-3], [97-3, 118-3], [98-3, 119-3], [99-3, 120-3], [100-3, 121-3],
        [101-3, 122-3], [102-3, 123-3], [103-3, 124-3], [104-3, 125-3], [105-3, 126-3], [106-3, 127-3], [107-3, 128-3],
        [108-3, 129-3], [109-3, 130-3], [110-3, 131-3], [111-3, 132-3], [112-3, 133-3], [113-3, 134-3], [114-3, 135-3]]
