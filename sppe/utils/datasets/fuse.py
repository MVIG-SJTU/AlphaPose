import os
from functools import reduce

import h5py
import numpy as np
from mxnet import context, nd
from mxnet.gluon.data import dataset

from opt import opt

from sppe.utils.pose import generateSampleBox


class Fuse(dataset.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        # root image folders
        self.img_folder = './data/'
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,             # COCO
                        9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23,             # MPII
                        28, 29, 32, 33)
        self.flipRef = ((2, 3), (4, 5), (6, 7),             # COCO
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17),
                        (18, 23), (19, 22), (20, 21),       # MPII
                        (28, 33), (29, 32), (30, 31))

        # create train/val split
        # COCO
        with h5py.File('./data/coco/annot_clean.h5', 'r') as annot:
            # train
            self.imgname_coco_train = annot['imgname'][:-5887]
            self.bndbox_coco_train = annot['bndbox'][:-5887]
            self.part_coco_train = annot['part'][:-5887]
            # val
            self.imgname_coco_val = annot['imgname'][-5887:]
            self.bndbox_coco_val = annot['bndbox'][-5887:]
            self.part_coco_val = annot['part'][-5887:]
        # AIC
        with h5py.File('./data/aic/annot_clean_hard.h5', 'r') as annot:
            # train
            self.imgname_aic_train = annot['imgname'][:]
            self.bndbox_aic_train = annot['bndbox'][:]
            self.part_aic_train = annot['part'][:]
        # MPII
        with h5py.File('./data/mpii/annot_mpii.h5', 'r') as annot:
            # train
            self.imgname_mpii_train = annot['imgname'][:]
            self.bndbox_mpii_train = annot['bndbox'][:]
            self.part_mpii_train = annot['part'][:]

        self.size_coco_train = self.imgname_coco_train.shape[0]
        self.size_aic_train = self.imgname_aic_train.shape[0]
        self.size_mpii_train = self.imgname_mpii_train.shape[0]

        self.size_train = self.size_coco_train + self.size_aic_train + self.size_mpii_train
        self.size_val = self.imgname_coco_val.shape[0]

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train and index < self.size_coco_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
            imgset = 'coco'
        elif self.is_train and index < self.size_coco_train + self.size_aic_train:
            baseIndex = self.size_coco_train
            part = self.part_aic_train[index - baseIndex]
            bndbox = self.bndbox_aic_train[index - baseIndex]
            imgname = self.imgname_aic_train[index - baseIndex]
            imgset = 'aic'
        elif self.is_train:
            baseIndex = self.size_coco_train + self.size_aic_train
            part = self.part_mpii_train[index - baseIndex]
            bndbox = self.bndbox_mpii_train[index - baseIndex]
            imgname = self.imgname_mpii_train[index - baseIndex]
            imgset = 'mpii'
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]
            imgset = 'coco'

        if imgset == 'coco':
            imgname = reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), imgname))
        elif imgset == 'mpii':
            imgname = reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), imgname))[:13]
        elif imgset == 'aic':
            imgname = reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), imgname))[:44]

        img_path = os.path.join(self.img_folder, imgset, 'images', imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     imgset, sf, self, train=self.is_train)

        inp, out, setMask, ori_inp = metaData

        return inp, out, setMask, ori_inp

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val


def default_mp_batchify_fn(data):
    """Collate data into batch. Use shared memory for stacking."""
    if isinstance(data[0], nd.NDArray):
        out = nd.empty((len(data),) + data[0].shape, dtype=data[0].dtype,
                       ctx=context.Context('cpu_shared', 0))
        return nd.stack(*data, out=out)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_mp_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
