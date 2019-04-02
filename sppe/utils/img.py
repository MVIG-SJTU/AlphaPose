import cv2
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon.data.vision import transforms

from opt import opt


def ToTensor(img):
    '''
    img: (h, w, 3), ranging from 0 to 255
    img_tensor: (3, h, w), normalized to [0, 1)
    '''
    img_tensor = transforms.ToTensor()(nd.array(img))
    # assert img_tensor.shape[0] == 3
    return img_tensor


def ToImg(tensor):
    '''
    tensor: (3, h, w)
    output: (h, w, 3)
    '''
    img = (np.transpose(tensor, (1, 2, 0)) * 255).copy()
    # assert img.shape[-1] == 3, img.shape
    return img


def cv_cropBox(img, ul, br, resH, resW):
    ul = ul
    br = (br - 1)
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.ndim == 2:
        img = img[np.newaxis, :]

    box_shape = [br[1] - ul[1], br[0] - ul[0]]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    img[:, :ul[1], :], img[:, :, :ul[0]] = 0, 0
    img[:, br[1] + 1:, :], img[:, :, br[0] + 1:] = 0, 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array(
        [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array(
        [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(ToImg(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return ToTensor(dst_img).asnumpy()


def cv_rotate(img, rot, resW, resH):

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

    dst_img = cv2.warpAffine(ToImg(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return ToTensor(dst_img).asnumpy()


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def flip(x):
    assert (x.ndim == 3 or x.ndim == 4)
    if x.ndim == 3:
        x = x[:, :, ::-1]
    elif x.ndim == 4:
        x = x[:, :, :, ::-1]
    return x


def shuffleLR(x, dataset):
    flipRef = dataset.flipRef
    assert (x.ndim == 3 or x.ndim == 4)
    for pair in flipRef:
        dim0, dim1 = pair
        dim0 -= 1
        dim1 -= 1
        idx = np.array((dim0, dim1))
        inv_idx = np.array((dim1, dim0))
        if x.ndim == 4:
            x[:, idx] = x[:, inv_idx]
        else:
            x[idx] = x[inv_idx]
    return x


def transformBox(pt, ul, br, inpH, inpW, resH, resW):
    center = np.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = np.zeros(2)
    _pt[0] = pt[0] - ul[0]
    _pt[1] = pt[1] - ul[1]
    # Move to center
    _pt[0] = _pt[0] + max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] + max(0, (lenH - 1) / 2 - center[1])
    pt = (_pt * resH) / lenH
    pt[0] = round(float(pt[0]))
    pt[1] = round(float(pt[1]))
    return pt


def transformBoxInvert(pt, ul, br, inpH, inpW, resH, resW):
    # type: (Tensor, Tensor, Tensor, float, float, float, float) -> Tensor

    center = nd.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = (pt * lenH) / resH

    if bool(((lenW - 1) / 2 - center[0]) > 0):
        _pt[0] = _pt[0] - ((lenW - 1) / 2 - center[0])
    if bool(((lenH - 1) / 2 - center[1]) > 0):
        _pt[1] = _pt[1] - ((lenH - 1) / 2 - center[1])

    new_point = nd.zeros(2)
    new_point[0] = _pt[0] + ul[0]
    new_point[1] = _pt[1] + ul[1]
    return new_point


def drawGaussian(img, pt, sigma, sig=1):
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))

    if sig < 0:
        g *= opt.spRate
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, scale=1.15, ctx=mx.cpu(), thr=0.2, rescale=None):
    '''
    img: (h, w, 3)
    '''
    L = class_IDs.shape[1]
    upscale_bbox = []
    final_bbox = []
    final_bbox_scores = []
    final_ids = []
    for i in range(L):
        if class_IDs[0][i].asscalar() != 0:
            continue
        if scores[0][i].asscalar() < thr:
            continue
        bbox = bounding_boxs[0][i].asnumpy()
        if rescale is not None:
            bbox = rescale_bbox_fn(bbox, rescale)
        upscale_bbox.append(upscale_bbox_fn(bbox.tolist(), img, scale=scale))
        final_bbox.append(bbox.tolist())
        final_bbox_scores.append(scores[0][i].asnumpy().tolist())
        final_ids.append(class_IDs[0][i].asnumpy().tolist())

    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(img, upscale_bbox)
        pose_input = pose_input.as_in_context(ctx)
    else:
        pose_input = None
    return pose_input, upscale_bbox, final_bbox_scores, final_ids, final_bbox


def rescale_bbox_fn(bbox, rescale):
    h_scale, w_scale = rescale
    bbox[0] /= w_scale
    bbox[2] /= w_scale
    bbox[1] /= h_scale
    bbox[3] /= h_scale
    return bbox


def upscale_bbox_fn(bbox, img, scale=1.15):
    new_bbox = []
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    w = (x1 - x0) / 2
    h = (y1 - y0) / 2
    center = [x0 + w, y0 + h]
    new_x0 = max(center[0] - w * scale, 0)
    new_y0 = max(center[1] - h * scale, 0)
    new_x1 = min(center[0] + w * scale, img.shape[1])
    new_y1 = min(center[1] + h * scale, img.shape[0])
    new_bbox = [new_x0, new_y0, new_x1, new_y1]
    return new_bbox


def crop_resize_normalize(img, bbox_list):
    img = ToTensor(img).asnumpy()
    img[0] -= 0.406
    img[1] -= 0.457
    img[2] -= 0.480

    output_list = []
    inputResH, inputResW = opt.inputResH, opt.inputResW
    for bbox in bbox_list:
        x0 = max(int(bbox[0]), 0)
        y0 = max(int(bbox[1]), 0)
        x1 = max(min(int(bbox[2]), int(img.shape[2])), x0 + 5)
        y1 = max(min(int(bbox[3]), int(img.shape[1])), y0 + 5)

        upLeft = np.array([x0, y0])
        bottomRight = np.array([x1, y1])

        res_img = cv_cropBox(img.copy(), upLeft, bottomRight, inputResH, inputResW)

        output_list.append(nd.array(res_img))
    output_array = nd.stack(*output_list)
    return output_array


def heatmap_to_coord(heatmaps, bbox_list):
    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]
    coords, maxvals = get_max_pred(heatmaps)
    preds = nd.zeros_like(coords)

    for i, bbox in enumerate(bbox_list):
        ul = [bbox[0], bbox[1]]
        br = [bbox[2], bbox[3]]
        pts = coords[i]

        for k in range(17):
            pt = transformBoxInvert(pts[k], ul, br, heatmap_height, heatmap_width,
                                    heatmap_height, heatmap_width)
            preds[i][k] = pt

    return preds, maxvals


def get_max_pred(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = nd.argmax(heatmaps_reshaped, 2)
    maxvals = nd.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = nd.floor((preds[:, :, 1]) / width)

    pred_mask = nd.tile(nd.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
