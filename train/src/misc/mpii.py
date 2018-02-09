import scipy.io
import scipy.misc
import numpy as np

# Load in annotations
datadir = '../../data/mpii/'
annotpath = datadir + 'annot/mpii_human_pose_v1_u12_1.mat'
annot = scipy.io.loadmat(annotpath)
annot = annot['RELEASE']
nimages = annot['img_train'][0][0][0].shape[0]

# Part info
parts = ['rank', 'rkne', 'rhip',
         'lhip', 'lkne', 'lank',
         'pelv', 'thrx', 'neck', 'head',
         'rwri', 'relb', 'rsho',
         'lsho', 'lelb', 'lwri']
nparts = len(parts)

def imgpath(idx):
    # Path to image
    filename = str(annot['annolist'][0][0][0]['image'][idx][0]['name'][0][0])
    return datadir + '/images/' + filename

def loadimg(idx):
    # Load in image
    return scipy.misc.imread(imgpath(idx))

def numpeople(idx):
    # Get number of people present in image
    example = annot['annolist'][0][0][0]['annorect'][idx]
    if len(example) > 0:
        return len(example[0])
    else:
        return 0

def istrain(idx):
    # Return true if image is in training set
    return (annot['img_train'][0][0][0][idx] and
            annot['annolist'][0][0][0]['annorect'][idx].size > 0 and
            'annopoints' in annot['annolist'][0][0][0]['annorect'][idx].dtype.fields)

def location(idx, person):
    # Return center of person, and scale factor
    example = annot['annolist'][0][0][0]['annorect'][idx]
    if ((not example.dtype.fields is None) and
        'scale' in example.dtype.fields and
        example['scale'][0][person].size > 0 and
        example['objpos'][0][person].size > 0):
        scale = example['scale'][0][person][0][0]
        x = example['objpos'][0][person][0][0]['x'][0][0]
        y = example['objpos'][0][person][0][0]['y'][0][0]
        return np.array([x, y]), scale
    else:
        return [-1, -1], -1

def partinfo(idx, person, part):
    # Part location and visibility
    # This function can take either the part name or the index of the part
    if type(part) == type(''):
        part = parts.index(part)

    example = annot['annolist'][0][0][0]['annorect'][idx]
    if example['annopoints'][0][person].size > 0:
        parts_info = example['annopoints'][0][person][0][0][0][0]
        for i in xrange(len(parts_info)):
            if parts_info[i]['id'][0][0] == part:
                if 'is_visible' in parts_info.dtype.fields:
                    v = parts_info[i]['is_visible']
                    v = v[0][0] if len(v) > 0 else 1
                    if type(v) is unicode:
                        v = int(v)
                else:
                    v = 1
                return np.array([parts_info[i]['x'][0][0], parts_info[i]['y'][0][0]], int), v
        return np.zeros(2, int), 0
    return -np.ones(2, int), -1

def normalization(idx, person):
    # Get head height for distance normalization
    if istrain(idx):
        example = annot['annolist'][0][0][0]['annorect'][idx]
        x1, y1 = int(example['x1'][0][person][0][0]), int(example['y1'][0][person][0][0])
        x2, y2 = int(example['x2'][0][person][0][0]), int(example['y2'][0][person][0][0])
        diff = np.array([y2 - y1, x2 - x1], np.float)
        return np.linalg.norm(diff) * .6
    return -1

def torsoangle(idx, person):
    # Get angle from pelvis to thorax, 0 means the torso is up vertically
    pt1 = partinfo(idx, person, 'pelv')[0]
    pt2 = partinfo(idx, person, 'thrx')[0]
    if not (pt1[0] == 0 or pt2[0] == 0):
        return 90 + np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180. / np.pi
    else:
        return 0
