from itertools import count
from threading import Thread
from queue import Queue
import json

import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL


class FileDetectionLoader():
    def __init__(self, input_source, cfg, opt, queueSize=128):
        self.cfg = cfg
        self.opt = opt
        self.bbox_file = input_source

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False)
        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
            # TODO: new features
            from easydict import EasyDict as edict
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2)
            })
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2,2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False, gpu_device=self.device,
                loss_type=cfg.LOSS['TYPE'])

        # initialize the det file list        
        boxes = None
        if isinstance(self.bbox_file,list):
            boxes = self.bbox_file
        else:
            with open(self.bbox_file, 'r') as f:
                boxes = json.load(f)
            assert boxes is not None, 'Load %s fail!' % self.bbox_file

        self.all_imgs = []
        self.all_boxes = {}
        self.all_scores = {}
        self.all_ids = {}
        num_boxes = 0
        for k_img in range(0, len(boxes)):
            det_res = boxes[k_img]
            img_name = det_res['image_id']
            if img_name not in self.all_imgs:
                self.all_imgs.append(img_name)
                self.all_boxes[img_name] = []
                self.all_scores[img_name] = []
                self.all_ids[img_name] = []
            x1, y1, w, h = det_res['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            score = det_res['score']
            self.all_boxes[img_name].append(bbox)
            self.all_scores[img_name].append(score)
            if 'idx' in det_res.keys():
                self.all_ids[img_name].append(int(det_res['idx']))
            else:
                self.all_ids[img_name].append(0)

        # initialize the queue used to store data
        """
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if opt.sp:
            self._stopped = False
            self.pose_queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value('b', False)
            self.pose_queue = mp.Queue(maxsize=queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        image_preprocess_worker = self.start_worker(self.get_detection)
        return [image_preprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)

    def wait_and_get(self, queue):
        if not self.stopped:
            return queue.get()

    def get_detection(self):
        
        for im_name_k in self.all_imgs:
            boxes = torch.from_numpy(np.array(self.all_boxes[im_name_k]))
            scores = torch.from_numpy(np.array(self.all_scores[im_name_k]))
            ids = torch.from_numpy(np.array(self.all_ids[im_name_k]))
            orig_img_k = cv2.cvtColor(cv2.imread(im_name_k), cv2.COLOR_BGR2RGB) #scipy.misc.imread(im_name_k, mode='RGB') is depreciated


            inps = torch.zeros(boxes.size(0), 3, *self._input_size)
            cropped_boxes = torch.zeros(boxes.size(0), 4)
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img_k, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            
            self.wait_and_put(self.pose_queue, (inps, orig_img_k, im_name_k, boxes, scores, ids, cropped_boxes))
        
        self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
        return
        
    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value
    @property
    def length(self):
        return len(self.all_imgs)

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
