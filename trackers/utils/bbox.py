import numpy as np
import cv2


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes = np.asarray(boxes)
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.copy(boxes)
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def clip_box(bbox, im_shape):
    h, w = im_shape[:2]
    bbox = np.copy(bbox)
    bbox[0] = max(min(bbox[0], w - 1), 0)
    bbox[1] = max(min(bbox[1], h - 1), 0)
    bbox[2] = max(min(bbox[2], w - 1), 0)
    bbox[3] = max(min(bbox[3], h - 1), 0)

    return bbox


def int_box(box):
    box = np.asarray(box, dtype=np.float)
    box = np.round(box)
    return np.asarray(box, dtype=np.int)


# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


def get_color(indx, cls_num=1):
    if indx >= cls_num:
        return (23 * indx % 255, 47 * indx % 255, 137 * indx % 255)
    base = int(np.ceil(pow(cls_num, 1. / 3)))
    return _to_color(indx, base)


def draw_detection(im, bboxes, scores=None, cls_inds=None, cls_name=None):
    # draw image
    bboxes = np.round(bboxes).astype(np.int)
    if cls_inds is not None:
        cls_inds = cls_inds.astype(np.int)
    cls_num = len(cls_name) if cls_name is not None else 2

    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        cls_indx = cls_inds[i] if cls_inds is not None else 1
        color = get_color(cls_indx, cls_num)

        thick = int((h + w) / 600)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)

        if cls_indx is not None:
            score = scores[i] if scores is not None else 1
            name = cls_name[cls_indx] if cls_name is not None else str(cls_indx)
            mess = '%s: %.3f' % (name, score) if cls_inds is not None else '%.3f' % (score, )
            cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                        0, 1e-3 * h, color, thick // 3)

    return imgcv
def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox
def transformBoxInvert(pt, bbox, resH, resW):
    center = torch.zeros(2)
    center[0] = (bbox[2] - 1 - bbox[0]) / 2
    center[1] = (bbox[3] - 1 - bbox[1]) / 2

    lenH = max(bbox[3] - bbox[1], (bbox[2] - bbox[0]) * resH / resW)
    lenW = lenH * resW / resH

    _pt = (pt * lenH) / resH

    if bool(((lenW - 1) / 2 - center[0]) > 0):
        _pt[0] = _pt[0] - ((lenW - 1) / 2 - center[0]).item()
    if bool(((lenH - 1) / 2 - center[1]) > 0):
        _pt[1] = _pt[1] - ((lenH - 1) / 2 - center[1]).item()

    new_point = torch.zeros(2)
    new_point[0] = _pt[0] + bbox[0]
    new_point[1] = _pt[1] + bbox[1]
    return new_point
