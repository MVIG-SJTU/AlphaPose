import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian'):
        self.img_folder = '../data/'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = 320
        self.inputResW = 256
        self.outputResH = 80
        self.outputResW = 64
        self.sigma = sigma
        self.scale_factor = (0.2, 0.3)
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

        '''
        Create train/val split
        '''
        # COCO
        with h5py.File('../data/coco/annot_clean.h5', 'r') as annot:
            # train
            self.imgname_coco_train = annot['imgname'][:-5887]
            self.bndbox_coco_train = annot['bndbox'][:-5887]
            self.part_coco_train = annot['part'][:-5887]
            # val
            self.imgname_coco_val = annot['imgname'][-5887:]
            self.bndbox_coco_val = annot['bndbox'][-5887:]
            self.part_coco_val = annot['part'][-5887:]
        # MPII
        with h5py.File('../data/mpii/annot_mpii.h5', 'r') as annot:
            # train
            self.imgname_mpii_train = annot['imgname'][:-1358]
            self.bndbox_mpii_train = annot['bndbox'][:-1358]
            self.part_mpii_train = annot['part'][:-1358]
            # val
            self.imgname_mpii_val = annot['imgname'][-1358:]
            self.bndbox_mpii_val = annot['bndbox'][-1358:]
            self.part_mpii_val = annot['part'][-1358:]

        self.size_coco_train = self.imgname_coco_train.shape[0]
        self.size_coco_val = self.imgname_coco_val.shape[0]
        self.size_train = self.imgname_coco_train.shape[0] + self.imgname_mpii_train.shape[0]
        self.size_val = self.imgname_coco_val.shape[0] + self.imgname_mpii_val.shape[0]
        self.train, self.valid = [], []

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train and index < self.size_coco_train:  # COCO
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
            imgset = 'coco'
        elif self.is_train:  # MPII
            part = self.part_mpii_train[index - self.size_coco_train]
            bndbox = self.bndbox_mpii_train[index - self.size_coco_train]
            imgname = self.imgname_mpii_train[index - self.size_coco_train]
            imgset = 'mpii'
        elif index < self.size_coco_val:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]
            imgset = 'coco'
        else:
            part = self.part_mpii_val[index - self.size_coco_val]
            bndbox = self.bndbox_mpii_val[index - self.size_coco_val]
            imgname = self.imgname_mpii_val[index - self.size_coco_val]
            imgset = 'mpii'

        if imgset == 'coco':
            imgname = reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), imgname))
        else:
            imgname = reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), imgname))[:13]

        img_path = os.path.join(self.img_folder, imgset, 'images', imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     imgset, sf, self, train=self.is_train)

        inp, out_bigcircle, out_smallcircle, out, setMask = metaData

        label = []
        for i in range(opt.nStack):
            if i < 2:
                # label.append(out_bigcircle.clone())
                label.append(out.clone())
            elif i < 4:
                # label.append(out_smallcircle.clone())
                label.append(out.clone())
            else:
                label.append(out.clone())

        return inp, label, setMask, imgset

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
