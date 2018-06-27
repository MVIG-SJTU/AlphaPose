import torch
import re
import os
import collections
from torch._six import string_classes, int_classes
import cv2
from opt import opt
from tqdm import tqdm


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = True


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def collate_fn_list(batch):
    img, inp, im_name = zip(*batch)
    img = collate_fn(img)
    im_name = collate_fn(im_name)

    return img, inp, im_name


def vis_res(final_result, outputpath, format='coco'):
    '''
    final_result: result dict of predictions
    outputpath: output directory
    format: coco or mpii
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                   YELLOW, ORANGE, PINK, RED, PINK, RED, PINK, RED]
    else:
        NotImplementedError

    for im_res in (final_result):
        im_name = im_res['imgname'].split('/')[-1]
        img = cv2.imread(os.path.join(opt.imgpath, im_name))
        for human in im_res['result']:
            part_line = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            # Draw keypoints
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= 0.3:
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                part_line[n] = (cor_x, cor_y)
                cv2.circle(img, (cor_x, cor_y), 6, p_color[n], -1)
            # Draw limbs
            for start_p, end_p in l_pair:
                if start_p in part_line and end_p in part_line:
                    start_p = part_line[start_p]
                    end_p = part_line[end_p]
                    cv2.line(img, start_p, end_p, YELLOW, 2)

        cv2.imwrite(os.path.join(outputpath, 'res' + im_name), img)
