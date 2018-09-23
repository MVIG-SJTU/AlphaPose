# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
import torch
import torch.utils.data as data
from utils.img import (load_image, cropBox)
from opt import opt


class Mscoco_minival(data.Dataset):
    def __init__(self, annoSet='coco-minival-images-newnms/test-dev'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.annot = dict()

        # Read in annotation information from hdf5 file
        tags = ['xmin', 'ymin', 'xmax', 'ymax']
        with h5py.File('./predict/annot/' + annoSet + '.h5', 'r') as a:
            for tag in tags:
                self.annot[tag] = a[tag][:]

        # Load in image file names
        with open('./predict/annot/' + annoSet + '_images.txt', 'r') as f:
            self.images = f.readlines()
        self.images = list(map(lambda x: x.strip('\n'), self.images))
        assert len(self.images) == self.annot['xmin'].shape[0]
        self.size = len(self.images)

        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))
        self.year = 2017

    def __getitem__(self, index):
        if self.year == 2014:
            imgname = self.images[index]
        else:
            imgname = self.images[index].split('_')[2]

        img_path = os.path.join(self.img_folder, imgname)
        img = load_image(img_path)

        ori_img = img.clone()
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        imght = img.size(1)
        imgwidth = img.size(2)
        upLeft = torch.Tensor(
            (float(self.annot['xmin'][index]), float(self.annot['ymin'][index])))
        bottomRight = torch.Tensor(
            (float(self.annot['xmax'][index]), float(self.annot['ymax'][index])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        inp = cropBox(img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
        ori_inp = cropBox(ori_img, upLeft, bottomRight,
                          opt.inputResH, opt.inputResW)
        metaData = (
            upLeft,
            bottomRight,
            ori_inp
        )
        box = torch.zeros(4)
        box[0] = upLeft[0]
        box[1] = upLeft[1]
        box[2] = bottomRight[0]
        box[3] = bottomRight[1]

        return inp, box, imgname, metaData

    def __len__(self):
        return self.size
