# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com), Haoyi Zhu
# -----------------------------------------------------

import platform
import random

import cv2
import numpy as np
import torch

from ..bbox import (_box_to_center_scale, _center_scale_to_box,
                    _clip_aspect_ratio)
from ..transforms import (addDPG, affine_transform, flip_joints_3d,
                          get_affine_transform, im_to_torch)

# Only windows visual studio 2013 ~2017 support compile c/cuda extensions
# If you force to compile extension on Windows and ensure appropriate visual studio
# is intalled, you can try to use these ext_modules.
if platform.system() != 'Windows':
    from ..roi_align import RoIAlign


class SimpleTransform(object):
    """Generation of cropped input person and pose heatmaps from SimplePose.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, add_dpg,
                 input_size, output_size, rot, sigma,
                 train, gpu_device=None, loss_type='MSELoss'):
        self._joint_pairs = dataset.joint_pairs
        self._scale_factor = scale_factor
        self._rot = rot
        self._add_dpg = add_dpg
        self._gpu_device = gpu_device

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids
        if platform.system() != 'Windows':
            self.roi_align = RoIAlign(self._input_size, sample_num=-1)
            if gpu_device is not None:
                self.roi_align = self.roi_align.to(gpu_device)

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

    def align_transform(self, image, boxes):
        """
        Performs Region of Interest (RoI) Align operator described in Mask R-CNN

        Arguments:
            input (ndarray [H, W, 3]): input images
            boxes (Tensor[K, 4]): the box coordinates in (x1, y1, x2, y2)
                format where the regions will be taken from.

        Returns:
            cropped_img (Tensor[K, C, output_size[0], output_size[1]])
            boxes (Tensor[K, 4]): new box coordinates
        """
        tensor_img = im_to_torch(image)
        tensor_img[0].add_(-0.406)
        tensor_img[1].add_(-0.457)
        tensor_img[2].add_(-0.480)

        new_boxes = _clip_aspect_ratio(boxes, self._aspect_ratio)
        cropped_img = self.roi_align(tensor_img.unsqueeze(0).to(self._gpu_device), new_boxes.to(self._gpu_device))
        return cropped_img, new_boxes[:, 1:]

    def _target_generator(self, joints_3d, num_joints):
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]),
                          dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(num_joints):
            mu_x = int(joints_3d[i, 0, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_3d[i, 1, 0] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)

    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        if num_joints == 136:
            target_weight[:26, :] = target_weight[:26, :] * 2
        elif num_joints == 133:
            target_weight[:23, :] = target_weight[:23, :] * 2
        elif num_joints == 68:
            target_weight[:26, :] = target_weight[:26, :] * 2

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        bbox = list(label['bbox'])
        gt_joints = label['joints_3d']

        imgwidth, imght = label['width'], label['height']
        assert imgwidth == src.shape[1] and imght == src.shape[0]
        self.num_joints = gt_joints.shape[0]

        joints_vis = np.zeros((self.num_joints, 1), dtype=np.float32)
        joints_vis[:, 0] = gt_joints[:, 0, 1]

        input_size = self._input_size

        if self._add_dpg and self._train:
            bbox = addDPG(bbox, imgwidth, imght)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)

        # half body transform
        if self._train and (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
            c_half_body, s_half_body = self.half_body_transform(
                gt_joints[:, :, 0], joints_vis
            )

            if c_half_body is not None and s_half_body is not None:
                center, scale = c_half_body, s_half_body

        # rescale
        if self._train:
            sf = self._scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0

        # rotation
        if self._train:
            rf = self._rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        else:
            r = 0

        joints = gt_joints
        if random.random() > 0.5 and self._train:
            # src, fliped = random_flip_image(src, px=0.5, py=0)
            # if fliped[0]:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints = flip_joints_3d(joints, imgwidth, self._joint_pairs)
            center[0] = imgwidth - center[0] - 1

        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        # deal with joints visibility
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        # generate training targets
        if self._loss_type == 'MSELoss':
            target, target_weight = self._target_generator(joints, self.num_joints)
        elif 'JointRegression' in self._loss_type:
            target, target_weight = self._integral_target_generator(joints, self.num_joints, inp_h, inp_w)
        elif self._loss_type == 'Combined':
            if self.num_joints == 68:
                hand_face_num = 42
            else:
                hand_face_num = 110
            target_mse, target_weight_mse = self._target_generator(joints[:-hand_face_num,:,:], self.num_joints-hand_face_num)
            target_inter, target_weight_inter = self._integral_target_generator(joints[-hand_face_num:,:,:], hand_face_num, inp_h, inp_w)

        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)
        
        if self._loss_type == 'Combined':
        	return img, [torch.from_numpy(target_mse), torch.from_numpy(target_inter)], [torch.from_numpy(target_weight_mse), torch.from_numpy(target_weight_inter)], torch.Tensor(bbox)
        else:
            return img, torch.from_numpy(target), torch.from_numpy(target_weight), torch.Tensor(bbox)

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale
