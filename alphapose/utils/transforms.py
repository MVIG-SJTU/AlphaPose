# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

"""Pose related transforrmation functions."""

import random

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.randn(1)[0] * x))


def box_transform(bbox, sf, imgwidth, imght, train):
    """Random scaling."""
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]
    if train:
        scaleRate = 0.25 * np.clip(np.random.randn() * sf, - sf, sf)

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, bbox[2] + width * scaleRate / 2)
        bbox[3] = min(imght, bbox[3] + ht * scaleRate / 2)
    else:
        scaleRate = 0.25

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, max(bbox[2] + width * scaleRate / 2, bbox[0] + 5))
        bbox[3] = min(imght, max(bbox[3] + ht * scaleRate / 2, bbox[1] + 5))

    return bbox


def addDPG(bbox, imgwidth, imght):
    """Add dpg for data augmentation, including random crop and random sample."""
    PatchScale = random.uniform(0, 1)
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]

    if PatchScale > 0.85:
        ratio = ht / width
        if (width < ht):
            patchWidth = PatchScale * width
            patchHt = patchWidth * ratio
        else:
            patchHt = PatchScale * ht
            patchWidth = patchHt / ratio

        xmin = bbox[0] + random.uniform(0, 1) * (width - patchWidth)
        ymin = bbox[1] + random.uniform(0, 1) * (ht - patchHt)
        xmax = xmin + patchWidth + 1
        ymax = ymin + patchHt + 1
    else:
        xmin = max(1, min(bbox[0] + np.random.normal(-0.0142, 0.1158) * width, imgwidth - 3))
        ymin = max(1, min(bbox[1] + np.random.normal(0.0043, 0.068) * ht, imght - 3))
        xmax = min(max(xmin + 2, bbox[2] + np.random.normal(0.0154, 0.1337) * width), imgwidth - 3)
        ymax = min(max(ymin + 2, bbox[3] + np.random.normal(-0.0013, 0.0711) * ht), imght - 3)

    bbox[0] = xmin
    bbox[1] = ymin
    bbox[2] = xmax
    bbox[3] = ymax

    return bbox


def im_to_torch(img):
    """Transform ndarray image to torch tensor.
    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.
    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    """Transform torch tensor to ndarray image.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.
    """
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))#scipy.misc.imread(img_path, mode='RGB'))


def to_numpy(tensor):
    # torch.Tensor => numpy.ndarray
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def cv_cropBox(img, bbox, input_size):
    """Crop bbox from image by Affinetransform.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.
    """
    xmin, ymin, xmax, ymax = bbox
    xmax -= 1
    ymax -= 1
    resH, resW = input_size

    lenH = max((ymax - ymin), (xmax - xmin) * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :, :]

    box_shape = [ymax - ymin, xmax - xmin]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    img[:, :ymin, :], img[:, :, :xmin] = 0, 0
    img[:, ymax + 1:, :], img[:, :, xmax + 1:] = 0, 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array([xmin - pad_size[1], ymin - pad_size[0]], np.float32)
    src[1, :] = np.array([xmax + pad_size[1], ymax + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 2:
        dst_img = dst_img[:, :, np.newaxis]

    return im_to_torch(torch.Tensor(dst_img))


def cv_cropBox_rot(img, bbox, input_size, rot):
    """Crop bbox from image by Affinetransform.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.
    """
    xmin, ymin, xmax, ymax = bbox
    xmax -= 1
    ymax -= 1
    resH, resW = input_size
    rot_rad = np.pi * rot / 180

    if img.dim() == 2:
        img = img[np.newaxis, :, :]

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2])

    src_dir = get_dir([0, (ymax - ymin) * -0.5], rot_rad)
    dst_dir = np.array([0, (resH - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
    dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 2:
        dst_img = dst_img[:, :, np.newaxis]

    return im_to_torch(torch.Tensor(dst_img))


def fix_cropBox(img, bbox, input_size):
    """Crop bbox from image by Affinetransform.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.
    """
    xmin, ymin, xmax, ymax = bbox
    input_ratio = input_size[0] / input_size[1]
    bbox_ratio = (ymax - ymin) / (xmax - xmin)
    if bbox_ratio > input_ratio:
        # expand width
        cx = (xmax + xmin) / 2
        h = ymax - ymin
        w = h / input_ratio
        xmin = cx - w / 2
        xmax = cx + w / 2
    elif bbox_ratio < input_ratio:
        # expand height
        cy = (ymax + ymin) / 2
        w = xmax - xmin
        h = w * input_ratio
        ymin = cy - h / 2
        ymax = cy + h / 2
    bbox = [int(x) for x in [xmin, ymin, xmax, ymax]]

    return cv_cropBox(img, bbox, input_size), bbox


def fix_cropBox_rot(img, bbox, input_size, rot):
    """Crop bbox from image by Affinetransform.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.
    """
    xmin, ymin, xmax, ymax = bbox
    input_ratio = input_size[0] / input_size[1]
    bbox_ratio = (ymax - ymin) / (xmax - xmin)
    if bbox_ratio > input_ratio:
        # expand width
        cx = (xmax + xmin) / 2
        h = ymax - ymin
        w = h / input_ratio
        xmin = cx - w / 2
        xmax = cx + w / 2
    elif bbox_ratio < input_ratio:
        # expand height
        cy = (ymax + ymin) / 2
        w = xmax - xmin
        h = w * input_ratio
        ymin = cy - h / 2
        ymax = cy + h / 2
    bbox = [int(x) for x in [xmin, ymin, xmax, ymax]]

    return cv_cropBox_rot(img, bbox, input_size, rot), bbox


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def cv_cropBoxInverse(inp, bbox, img_size, output_size):
    """Paste the cropped bbox to the original image.
    Parameters
    ----------
    inp: torch.Tensor
        A tensor with shape: `(3, height, width)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    img_size: tuple
        Original image size, as (img_H, img_W).
    output_size: tuple
        Cropped input size, as (height, width).
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, img_H, img_W)`.
    """
    xmin, ymin, xmax, ymax = bbox
    xmax -= 1
    ymax -= 1
    resH, resW = output_size
    imgH, imgW = img_size

    lenH = max((ymax - ymin), (xmax - xmin) * resH / resW)
    lenW = lenH * resW / resH
    if inp.dim() == 2:
        inp = inp[np.newaxis, :, :]

    box_shape = [ymax - ymin, xmax - xmin]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = 0
    src[1, :] = np.array([resW - 1, resH - 1], np.float32)
    dst[0, :] = np.array([xmin - pad_size[1], ymin - pad_size[0]], np.float32)
    dst[1, :] = np.array([xmax + pad_size[1], ymax + pad_size[0]], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(inp), trans,
                             (imgW, imgH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 3 and dst_img.shape[2] == 1:
        dst_img = dst_img[:, :, 0]
        return dst_img
    elif dst_img.ndim == 2:
        return dst_img
    else:
        return im_to_torch(torch.Tensor(dst_img))


def cv_rotate(img, rot, input_size):
    """Rotate image by Affinetransform.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    rot: int
        Rotation degree.
    input_size: tuple
        Resulting image size, as (height, width).
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.
    """
    resH, resW = input_size
    center = np.array((resW - 1, resH - 1)) / 2
    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (resH - 1) * -0.5], rot_rad)
    dst_dir = np.array([0, (resH - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
    dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 2:
        dst_img = dst_img[:, :, np.newaxis]

    return im_to_torch(torch.Tensor(dst_img))


def count_visible(bbox, joints_3d):
    """Count number of visible joints given bound box."""
    vis = np.logical_and.reduce((
        joints_3d[:, 0, 0] > 0,
        joints_3d[:, 0, 0] > bbox[0],
        joints_3d[:, 0, 0] < bbox[2],
        joints_3d[:, 1, 0] > 0,
        joints_3d[:, 1, 0] > bbox[1],
        joints_3d[:, 1, 0] < bbox[3],
        joints_3d[:, 0, 1] > 0,
        joints_3d[:, 1, 1] > 0
    ))
    return np.sum(vis), vis


def drawGaussian(img, pt, sigma):
    """Draw 2d gaussian on input image.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    pt: list or tuple
        A point: (x, y).
    sigma: int
        Sigma of gaussian distribution.
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.
    """
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def flip_heatmap(heatmap, joint_pairs, shift=False):
    """Flip pose heatmap according to joint pairs.
    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.
    Returns
    -------
    numpy.ndarray
        Flipped heatmap.
    """
    assert (heatmap.dim() == 3 or heatmap.dim() == 4)
    out = flip(heatmap)

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        if out.dim() == 4:
            out[:, idx] = out[:, inv_idx]
        else:
            out[idx] = out[inv_idx]

    if shift:
        if out.dim() == 3:
            out[:, :, 1:] = out[:, :, 0:-1]
        else:
            out[:, :, :, 1:] = out[:, :, :, 0:-1]
    return out


def flip_joints_3d(joints_3d, width, joint_pairs):
    """Flip 3d joints.
    Parameters
    ----------
    joints_3d : numpy.ndarray
        Joints in shape (num_joints, 3, 2)
    width : int
        Image width.
    joint_pairs : list
        List of joint pairs.
    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3, 2)
    """
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :, 0], joints[pair[1], :, 0] = \
            joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
        joints[pair[0], :, 1], joints[pair[1], :, 1] = \
            joints[pair[1], :, 1], joints[pair[0], :, 1].copy()

    joints[:, :, 0] *= joints[:, :, 1]
    return joints


def heatmap_to_coord_simple(hms, bbox, hms_flip=None, **kwargs):
    if hms_flip is not None:
        hms = (hms + hms_flip) / 2
    if not isinstance(hms,np.ndarray):
        hms = hms.cpu().data.numpy()
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                             hm[py + 1][px] - hm[py - 1][px]))
            coords[p] += np.sign(diff) * .25

    preds = np.zeros_like(coords)

    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale,
                                   [hm_w, hm_h])

    return preds, maxvals


def heatmap_to_coord_simple_regress(preds, bbox, hm_shape, norm_type, hms_flip=None):
    def integral_op(hm_1d):
        if hm_1d.device.index is not None:
            hm_1d = hm_1d * torch.cuda.comm.broadcast(torch.arange(hm_1d.shape[-1]).type(
                torch.cuda.FloatTensor), devices=[hm_1d.device.index])[0]
        else:
            hm_1d = hm_1d * torch.arange(hm_1d.shape[-1]).type(torch.FloatTensor)
        return hm_1d

    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
    hm_height, hm_width = hm_shape
    num_joints = preds.shape[1]

    pred_jts, pred_scores = _integral_tensor(preds, num_joints, False, hm_width, hm_height, 1, integral_op, norm_type)
    pred_jts = pred_jts.reshape(pred_jts.shape[0], num_joints, 2)

    if hms_flip is not None:
        if hms_flip.dim() == 3:
            hms_flip = hms_flip.unsqueeze(0)
        pred_jts_flip, pred_scores_flip = _integral_tensor(hms_flip, num_joints, False, hm_width, hm_height, 1, integral_op, norm_type)
        pred_jts_flip = pred_jts_flip.reshape(pred_jts_flip.shape[0], num_joints, 2)

        pred_jts = (pred_jts + pred_jts_flip) / 2
        pred_scores = (pred_scores + pred_scores_flip) / 2

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 3 or 4"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(np.float32)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(np.float32)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale,
                                               [hm_width, hm_height])

    if preds.shape[0] == 1:
        preds = preds[0]
        pred_scores = pred_scores[0]
    return preds, pred_scores


def _integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth, integral_operation, norm_type='softmax'):
    # normalization
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = norm_heatmap(norm_type, preds)

    # get heatmap confidence
    if norm_type == 'sigmoid':
        maxvals, _ = torch.max(preds, dim=2, keepdim=True)
    else:
        maxvals = torch.ones(
            (*preds.shape[:2], 1), dtype=torch.float, device=preds.device)

    # normalized to probability
    heatmaps = preds / preds.sum(dim=2, keepdim=True)
    heatmaps = heatmaps.reshape(
        (heatmaps.shape[0], num_joints, hm_depth, hm_height, hm_width))

    # The edge probability
    hm_x = heatmaps.sum((2, 3))
    hm_y = heatmaps.sum((2, 4))
    hm_z = heatmaps.sum((3, 4))

    hm_x = integral_operation(hm_x)
    hm_y = integral_operation(hm_y)
    hm_z = integral_operation(hm_z)

    coord_x = hm_x.sum(dim=2, keepdim=True)
    coord_y = hm_y.sum(dim=2, keepdim=True)
    coord_z = hm_z.sum(dim=2, keepdim=True)

    coord_x = coord_x / float(hm_width) - 0.5
    coord_y = coord_y / float(hm_height) - 0.5
    if output_3d:
        coord_z = coord_z / float(hm_depth) - 0.5
        pred_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 3))
    else:
        pred_jts = torch.cat((coord_x, coord_y), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))
    return pred_jts, maxvals.float()


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sigmoid':
        return heatmap.sigmoid()
    elif norm_type == 'divide_sum':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = heatmap / heatmap.sum(dim=2, keepdim=True)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_func_heatmap_to_coord(cfg):
    if cfg.DATA_PRESET.TYPE == 'simple':
        if cfg.LOSS.TYPE == 'MSELoss':
            return heatmap_to_coord_simple
        elif cfg.LOSS.TYPE == 'L1JointRegression':
            return heatmap_to_coord_simple_regress
        elif cfg.LOSS.TYPE == 'Combined':
            return [heatmap_to_coord_simple, heatmap_to_coord_simple_regress]
    else:
        raise NotImplementedError
