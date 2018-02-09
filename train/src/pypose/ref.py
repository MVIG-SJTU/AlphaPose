import os
from scipy.misc import imread
import h5py

# Home of the posenet directory, (change if not in your home directory)
posedir = os.environ["HOME"] + '/posenet'
# Global options
in_res = [256, 256]
out_res = [64, 64]

# Load annotations
# Example call: ref.load('mpii','train')
def load(dataset, settype):
    return h5py.File('%s/data/%s/annot/%s.h5' % (posedir, dataset, settype), 'r+')

# Part reference
parts = {'flic':['lsho', 'lelb', 'lwri',
                 'rsho', 'relb', 'rwri',
                 'lhip', 'rhip',
                 'leye', 'reye', 'nose'],
         'mpii':['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']}

flipped_parts = {'flic':[3, 4, 5, 0, 1, 2, 7, 6, 9, 8, 10],
                 'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

part_pairs = {'flic':[[0, 3], [1, 4], [2, 5], [6, 7], [8, 9, 10]],
              'mpii':[[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

pair_names = {'flic':['shoulder', 'elbow', 'wrist', 'hip', 'face'],
              'mpii':['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}

def partinfo(annot, idx, part):
    # This function can take either the part name or the index of the part
    if type(part) is str:
        part = parts[annot.attrs['name']].index(part)
    return annot['part'][idx, part]

# Load in an image
def loadimg(annot, idx):
    imgpath = '%s/data/%s/images/%s' % (posedir, annot.attrs['name'], annot['imgname'][idx])
    return imread(imgpath)
