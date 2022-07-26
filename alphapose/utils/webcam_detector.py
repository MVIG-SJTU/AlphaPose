from itertools import count
from threading import Thread
from queue import Queue

import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL


class WebCamDetectionLoader():
    def __init__(self, input_source, detector, cfg, opt, queueSize=1):
        self.cfg = cfg
        self.opt = opt

        stream = cv2.VideoCapture(int(input_source))
        assert stream.isOpened(), 'Cannot capture source'
        self.path = input_source
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize}
        stream.release()

        self.detector = detector

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
        image_preprocess_worker = self.start_worker(self.frame_preprocess)
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

    def frame_preprocess(self):
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'

        # keep looping infinitely
        for i in count():
            if self.stopped:
                stream.release()
                return
            if not self.pose_queue.full():
                # otherwise, ensure the queue has room in it
                (grabbed, frame) = stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                    stream.release()
                    return

                # expected frame shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(frame)

                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)

                im_dim_list_k = frame.shape[1], frame.shape[0]

                orig_img = frame[:, :, ::-1]
                im_name = str(i) + '.jpg'
                # im_dim_list = im_dim_list_k

                with torch.no_grad():
                    # Record original image resolution
                    im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
                img_det = self.image_detection((img_k, orig_img, im_name, im_dim_list_k))
                self.image_postprocess(img_det)

    def image_detection(self, inputs):
        img, orig_img, im_name, im_dim_list = inputs
        if img is None or self.stopped:
            return (None, None, None, None, None, None, None)

        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return (orig_img, im_name, None, None, None, None, None)
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            if self.opt.tracking:
                ids = dets[:, 6:7]
            else:
                ids = torch.zeros(scores.shape)

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return (orig_img, im_name, None, None, None, None, None)
        inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes_k.size(0), 4)
        return (orig_img, im_name, boxes_k, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self, inputs):
        with torch.no_grad():
            (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = inputs
            if orig_img is None or self.stopped:
                self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                return
            if boxes is None or boxes.nelement() == 0:
                self.wait_and_put(self.pose_queue, (None, orig_img, im_name, boxes, scores, ids, None))
                return
            # imght = orig_img.shape[0]
            # imgwidth = orig_img.shape[1]
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)

            self.wait_and_put(self.pose_queue, (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes))

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
