import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt


class Mpii(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian'):
        self.img_folder = '../data/mpii/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = 320
        self.inputResW = 256
        self.outputResH = 80
        self.outputResW = 64
        self.sigma = sigma
        self.scale_factor = (0.2, 0.3)
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_mpii = 16
        self.nJoints = 16

        self.accIdxs = (1, 2, 3, 4, 5, 6,
                        11, 12, 15, 16)
        self.flipRef = ((1, 6), (2, 5), (3, 4),
                        (11, 16), (12, 15), (13, 14))

        # create train/val split
        with h5py.File('../data/mpii/annot_mpii.h5', 'r') as annot:
            # train
            self.imgname_mpii_train = annot['imgname'][:-1358]
            self.bndbox_mpii_train = annot['bndbox'][:-1358]
            self.part_mpii_train = annot['part'][:-1358]
            # val
            self.imgname_mpii_val = annot['imgname'][-1358:]
            self.bndbox_mpii_val = annot['bndbox'][-1358:]
            self.part_mpii_val = annot['part'][-1358:]

        self.size_train = self.imgname_mpii_train.shape[0]
        self.size_val = self.imgname_mpii_val.shape[0]
        self.train, self.valid = [], []

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_mpii_train[index]
            bndbox = self.bndbox_mpii_train[index]
            imgname = self.imgname_mpii_train[index]
        else:
            part = self.part_mpii_val[index]
            bndbox = self.bndbox_mpii_val[index]
            imgname = self.imgname_mpii_val[index]

        imgname = reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), imgname))[:13]
        img_path = os.path.join(self.img_folder, imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'mpii', sf, self, train=self.is_train)

        inp, out_bigcircle, out_smallcircle, out, setMask = metaData

        label = []
        for i in range(opt.nStack):
            if i < 2:
                #label.append(out_bigcircle.clone())
                label.append(out.clone())
            elif i < 4:
                #label.append(out_smallcircle.clone())
                label.append(out.clone())
            else:
                label.append(out.clone())

        return inp, label, setMask

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
