import numpy as np
import scipy.misc
import scipy.signal
import math

import draw
import ref

# =============================================================================
# General image processing functions
# =============================================================================

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    return scipy.misc.imresize(new_img, res)

def two_pt_crop(img, scale, pt1, pt2, pad, res, chg=None):
    center = (pt1+pt2) / 2
    scale = max(20*scale, np.linalg.norm(pt1-pt2)) * .007
    scale *= pad
    angle = math.atan2(pt2[1]-pt1[1],pt2[0]-pt1[0]) * 180 / math.pi - 90
    flip = False

    # Handle data augmentation
    if chg is not None:
        # Flipping
        if 'flip' in chg:
            if np.random.rand() < .5:
                flip = True
        # Scaling
        if 'scale' in chg:
            scale *= min(1+chg['scale'], max(1-chg['scale'], (np.random.randn() * chg['scale']) + 1))
        # Rotation
        if 'rotate' in chg:
            angle += np.random.randint(-chg['rotate'], chg['rotate'] + 1)
        # Translation
        if 'translate' in chg:
            for i in xrange(2):
                offset = np.random.randint(-chg['translate'], chg['translate'] + 1) * scale
                center[i] += offset

    # Create input image
    cropped = crop(img, center, scale, res, rot=angle)
    inp = np.zeros((3, res[0], res[1]))
    for i in xrange(3):
        inp[i, :, :] = cropped[:, :, i]

    # Create heatmap
    hm = np.zeros((2,res[0],res[1]))
    draw.gaussian(hm[0],transform(pt1, center, scale, res, rot=angle), 2)
    draw.gaussian(hm[1],transform(pt2, center, scale, res, rot=angle), 2)

    if flip:
        inp = np.array([np.fliplr(inp[i]) for i in xrange(len(inp))])
        hm = np.array([np.fliplr(hm[i]) for i in xrange(len(hm))])

    return inp, hm

def nms(img):
    # Do non-maximum suppression on a 2D array
    win_size = 3
    domain = np.ones((win_size, win_size))
    maxes = scipy.signal.order_filter(img, domain, win_size ** 2 - 1)
    diff = maxes - img
    result = img.copy()
    result[diff > 0] = 0
    return result

# =============================================================================
# Helpful display functions
# =============================================================================

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x):
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color

def sample_with_heatmap(dataset, inp, out, num_rows=2, parts_to_show=None):
    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in xrange(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = np.ceil(float(len(parts_to_show)) / num_rows)
    size = img.shape[0] / num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        if type(part) is str:
            part_idx = ref.parts[dataset].index(part)
        else:
            part_idx = part
        out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = out_resized.astype(float)/255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img

def sample_with_skeleton(annot, idx, preds, res=None):

    # Load image and basic info
    ds = annot.attrs['name']
    img = ref.loadimg(annot, idx)
    c = annot['center'][idx]
    s = annot['scale'][idx]
    if res is None:
        res = [256, 256]

    # Skeleton colors
    colors = [(255, 0, 0),          # Upper arm (left)
              (255, 100, 100),      # Lower arm (left)
              (0, 0, 255),          # Upper arm (right)
              (100, 100, 255),      # Lower arm (right)
              (100, 255, 100),      # Head/neck/face
              (255, 75, 0),        # Upper leg (left)
              (255, 175, 100),      # Lower leg (left)
              (0, 75, 255),        # Upper leg (right)
              (100, 175, 255)       # Lower leg (right)
              ]

    # Draw arms
    draw.limb(img, preds[ref.parts[ds].index('lsho')], preds[ref.parts[ds].index('lelb')], colors[0], 5 * s)
    draw.limb(img, preds[ref.parts[ds].index('lwri')], preds[ref.parts[ds].index('lelb')], colors[1], 5 * s)
    draw.limb(img, preds[ref.parts[ds].index('rsho')], preds[ref.parts[ds].index('relb')], colors[2], 5 * s)
    draw.limb(img, preds[ref.parts[ds].index('rwri')], preds[ref.parts[ds].index('relb')], colors[3], 5 * s)

    if ds == 'mpii':
        # MPII
        # Draw head
        draw.circle(img, preds[ref.parts[ds].index('head')], colors[4], 5 * s)
        draw.circle(img, preds[ref.parts[ds].index('neck')], colors[4], 5 * s)

        # Draw legs
        draw.limb(img, preds[ref.parts[ds].index('lhip')], preds[ref.parts[ds].index('lkne')], colors[5], 5 * s)
        draw.limb(img, preds[ref.parts[ds].index('lank')], preds[ref.parts[ds].index('lkne')], colors[6], 5 * s)
        draw.limb(img, preds[ref.parts[ds].index('rhip')], preds[ref.parts[ds].index('rkne')], colors[7], 5 * s)
        draw.limb(img, preds[ref.parts[ds].index('rank')], preds[ref.parts[ds].index('rkne')], colors[8], 5 * s)

    elif ds == 'flic':
        # FLIC
        # Draw face
        draw.circle(img, preds[ref.parts[ds].index('leye')], colors[4], 3 * s)
        draw.circle(img, preds[ref.parts[ds].index('reye')], colors[4], 3 * s)
        draw.circle(img, preds[ref.parts[ds].index('nose')], colors[4], 3 * s)

        # Draw hips
        draw.circle(img, preds[ref.parts[ds].index('lhip')], colors[5], 5 * s)
        draw.circle(img, preds[ref.parts[ds].index('rhip')], colors[7], 5 * s)

    return crop(img, c, s, res)
