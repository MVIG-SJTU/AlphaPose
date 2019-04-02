import sys
import os
import time
import tqdm
import threading

import mxnet as mx
import gluoncv as gcv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sppe.models.sefastpose import FastPose_SE
from pose_utils import pose_nms

from opt import opt
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    import queue
# otherwise, import the Queue class for Python 2.7
else:
    import Queue as queue


# box conf = 0.05: 17.3FPS -> 19.3FPS
# box conf = 0.10: 20.0FPS -> 22.3FPS
# box conf = 0.20: 22.1FPS -> 24.8FPS
# box conf = 0.30: 23.0FPS -> 25.9FPS
# box conf = 0.40: 24.0FPS -> 26.6FPS
# box conf = 0.50: 24.5FPS -> 27.0FPS

ctx = mx.gpu()


class PoseStage:
    def __init__(self, target, queue_size=None, prev=None):
        self.target = target
        self.stopped = False
        self.prev = prev
        # initialize the queue used to store data
        if queue_size is not None:
            self.Q = queue.Queue(maxsize=queue_size)
        self.t = None

    def next(self, timeout=None):
        return self.Q.get(timeout=timeout)

    def start(self):
        self.t = threading.Thread(target=self.target, args=())
        self.t.daemon = True
        self.t.start()

    def wait_for_queue(self, time_step):
        while self.Q.full():
            time.sleep(time_step)

    def wait_for_stop(self, time_step):
        if self.prev is not None:
            while not self.prev.stopped:
                time.sleep(time_step)
        while not self.Q.empty():
            time.sleep(time_step)


class ImageLoader(PoseStage):
    '''Load images for prediction'''
    def __init__(self, im_names, batch_size=1, queue_size=16):
        super(ImageLoader, self).__init__(self.get_batch_image, queue_size)
        self.img_dir = opt.inputpath
        self.img_list = im_names
        self.batch_size = batch_size
        self.data_len = len(self.img_list)
        self.num_batches = (self.data_len + batch_size - 1) // batch_size

        for i in range(len(self.img_list)):
            self.img_list[i] = self.img_list[i].rstrip('\n').rstrip('\r')
            self.img_list[i] = os.path.join(opt.inputpath, self.img_list[i])

    def __len__(self):
        return len(self.img_list)

    def get_batch_image(self):
        time_rec = []
        tensor_size = int(opt.inp_dim)
        for i in range(self.num_batches):
            tic = time.time()
            tensor_batch = []
            img_batch = []
            img_size_batch = []
            begin_idx = i * self.batch_size
            end_idx = min(begin_idx + self.batch_size, self.data_len)
            img_name_list = self.img_list[begin_idx:end_idx]

            # laod images
            for k in range(begin_idx, end_idx):
                tensor_k, img_k, img_size_k = self.load_fn(self.img_list[k], tensor_size)
                tensor_batch.append(tensor_k)
                img_batch.append(img_k)
                img_size_batch.append(img_size_k)

            tensor_batch = mx.nd.concatenate(tensor_batch, axis=0)
            img_size_batch = mx.nd.array(img_size_batch, dtype='float32')
            img_size_batch = img_size_batch.tile(reps=[1, 2])

            toc = time.time()
            time_rec.append(toc - tic)

            # put into queue
            self.wait_for_queue(0.5)
            self.Q.put((tensor_batch, img_batch, img_size_batch, img_name_list))

        self.wait_for_stop(1)
        print('ImageLoader: %fs' % (np.mean(time_rec)))
        self.stopped = True

    def load_fn(self, img_name, tensor_size):
        '''
        Load single image from the given file
        INPUT:
            img_name: string, image file name
            tensor_size: int, image size after resize
        OUTPUT:
            tensor: mx.nd, input tensor for detection
            img: mx.nd, original image in nd type
            img_size: (int, int), original image size
        '''
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        img = mx.image.imread(img_name)
        img_size = (img.shape[1], img.shape[0])

        # resize image
        tensor = gcv.data.transforms.image.resize_long(img, tensor_size, interp=9)
        tensor = mx.nd.image.to_tensor(tensor)
        tensor = mx.nd.image.normalize(tensor, mean=mean, std=std)
        tensor = tensor.expand_dims(0)

        # pad tensor
        pad_h = tensor_size - tensor.shape[2]
        pad_w = tensor_size - tensor.shape[3]
        pad_shape = (0, 0, 0, 0, 0, pad_h, 0, pad_w)
        tensor = mx.nd.pad(tensor, mode='constant',
                           constant_value=0.5, pad_width=pad_shape)

        return tensor, img, img_size


class VideoLoader(PoseStage):
    '''Load video for prediction'''
    def __init__(self, path, batch_size=1, queue_size=16):
        super(VideoLoader, self).__init__(self.get_batch_image, queue_size)
        self.path = path
        self.batch_size = batch_size
        self.stream = cv2.VideoCapture(path)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.data_len = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_batches = (self.data_len + batch_size - 1) // batch_size

    def __len__(self):
        return self.data_len

    def video_info(self):
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return fourcc, fps, frame_size

    def get_batch_image(self):
        time_rec = []
        tensor_size = int(opt.inp_dim)
        for i in range(self.num_batches):
            tic = time.time()
            tensor_batch = []
            img_batch = []
            img_size_batch = []
            img_name_list = []
            begin_idx = i * self.batch_size
            end_idx = min(begin_idx + self.batch_size, self.data_len)

            # transform video frames
            for k in range(begin_idx, end_idx):
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stream.release()
                    self.wait_for_stop(1)
                    print('VideoLoader: %fs' % (np.mean(time_rec)))
                    self.stopped = True
                    return

                tensor_k, img_k, img_size_k = self.transform_fn(frame, tensor_size)
                tensor_batch.append(tensor_k)
                img_batch.append(img_k)
                img_size_batch.append(img_size_k)
                img_name_list.append(str(k) + '.jpg')

            tensor_batch = mx.nd.concatenate(tensor_batch, axis=0)
            img_size_batch = mx.nd.array(img_size_batch, dtype='float32')
            img_size_batch = img_size_batch.tile(reps=[1, 2])

            toc = time.time()
            time_rec.append(toc - tic)

            # put into queue
            self.wait_for_queue(0.5)
            self.Q.put((tensor_batch, img_batch, img_size_batch, img_name_list))

        self.wait_for_stop(1)
        print('VideoLoader: %fs' % (np.mean(time_rec)))
        self.stopped = True

    def transform_fn(self, frame, tensor_size):
        '''
        Transform single frame to input format
        INPUT:
            frame: np.nd, image file in bgr channels
            tensor_size: int, image size after resize
        OUTPUT:
            tensor: mx.nd, input tensor for detection
            img: mx.nd, original image in nd type
            img_size: (int, int), original image size
        '''
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        img = mx.nd.array(frame[:, :, ::-1], dtype='uint8')
        img_size = (img.shape[1], img.shape[0])

        # resize image
        tensor = gcv.data.transforms.image.resize_long(img, tensor_size, interp=9)
        tensor = mx.nd.image.to_tensor(tensor)
        tensor = mx.nd.image.normalize(tensor, mean=mean, std=std)
        tensor = tensor.expand_dims(0)

        # pad tensor
        pad_h = tensor_size - tensor.shape[2]
        pad_w = tensor_size - tensor.shape[3]
        pad_shape = (0, 0, 0, 0, 0, pad_h, 0, pad_w)
        tensor = mx.nd.pad(tensor, mode='constant',
                           constant_value=0.5, pad_width=pad_shape)

        return tensor, img, img_size


class Detector(PoseStage):
    '''Person detection'''
    def __init__(self, data_loader, queue_size=16):
        super(Detector, self).__init__(self.detect_fn, queue_size, prev=data_loader)
        self.input_size = int(opt.inp_dim)
        self.data_len = len(data_loader)
        self.num_batches = data_loader.num_batches

        # model config
        print('Loading yolo3_darknet53_coco ...')
        self.net = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
        self.net.set_nms(opt.nms_thresh, post_nms=-1)
        self.person_idx = self.net.classes.index('person')
        print('Modifying output layers to ignore non-person classes...')
        self.reset_class()
        self.net.collect_params().reset_ctx(ctx)
        self.net.hybridize()

    def __len__(self):
        return self.data_len

    def detect_fn(self):
        time_rec = []
        for _ in range(self.num_batches):
            if self.prev.stopped:
                break

            try:
                tensor_batch, img_batch, img_size_batch, img_name_list = self.prev.next(timeout=1)
            except Exception:
                continue

            tic = time.time()

            # get prediction
            class_idxs, scores, boxes = self.net(tensor_batch.copyto(ctx))
            class_idxs = class_idxs.copyto(mx.cpu())
            scores = scores.copyto(mx.cpu())
            boxes = boxes.copyto(mx.cpu())

            toc = time.time()
            time_rec.append(toc - tic)

            # put into queue
            for i in range(scores.shape[0]):
                self.wait_for_queue(0.1)
                self.Q.put((img_batch[i], boxes[i], class_idxs[i, :, 0], scores[i, :, 0], img_name_list[i], img_size_batch[i]))

        self.wait_for_stop(1)
        print('Detector: %fs' % (np.mean(time_rec)))
        self.stopped = True

    def reset_class(self):
        '''Modify output layers to ignore non-person classes'''
        self.net._clear_cached_op()
        len_per_anchor = 5 + len(self.net.classes)

        for output in self.net.yolo_outputs:
            num_anchors = output._num_anchors
            picked_channels = np.array(list(range(len_per_anchor)) * num_anchors)
            picked_channels = np.where((picked_channels < 5) |
                                       (picked_channels == 5 + self.person_idx))

            parameters = output.prediction.params
            for k in parameters:
                if 'weight' in k:
                    key_weight = k
                    init_weight = parameters[k].data()[picked_channels]
                    in_channels = parameters[k].data().shape[1]
                elif 'bias' in k:
                    key_bias = k
                    init_bias = parameters[k].data()[picked_channels]

            output.prediction = mx.gluon.nn.Conv2D(6 * num_anchors,
                                                   in_channels=in_channels,
                                                   kernel_size=1,
                                                   padding=0,
                                                   strides=1,
                                                   prefix=output.prediction.prefix)
            output.prediction.collect_params().initialize()
            output.prediction.params[key_weight].set_data(init_weight)
            output.prediction.params[key_bias].set_data(init_bias)
            output._classes = 1
            output._num_pred = 6
        self.net._classes = ['person']


class DetectionProcessor(PoseStage):
    '''Transform pose coordinates to box frames'''
    def __init__(self, detector, queue_size=50):
        super(DetectionProcessor, self).__init__(self.transform_fn, queue_size, prev=detector)
        self.data_len = len(detector)
        self.input_size = int(opt.inp_dim)
        self.person_idx = detector.person_idx

    def __len__(self):
        return self.data_len

    def transform_fn(self):
        time_rec = []
        for _ in range(self.data_len):
            if self.prev.stopped:
                break

            try:
                img, boxes, class_idxs, scores, img_name, img_size = self.prev.next(timeout=1)
            except Exception:
                continue

            tic = time.time()

            # rescale coordinates
            scaling_factor = mx.nd.min(self.input_size / img_size)
            boxes /= scaling_factor

            # cilp coordinates
            boxes[:, [0, 2]] = mx.nd.clip(boxes[:, [0, 2]], 0., img_size[0].asscalar() - 1)
            boxes[:, [1, 3]] = mx.nd.clip(boxes[:, [1, 3]], 0., img_size[1].asscalar() - 1)

            # select boxes
            mask1 = (class_idxs == self.person_idx).asnumpy()
            mask2 = (scores > opt.confidence).asnumpy()
            picked_idxs = np.where((mask1 + mask2) > 1)[0]

            # put into queue
            self.wait_for_queue(0.1)
            if picked_idxs.shape[0] == 0:
                self.Q.put((img, None, None, img_name))
            else:
                self.Q.put((img, boxes[picked_idxs], scores[picked_idxs], img_name))

            toc = time.time()
            time_rec.append(toc - tic)

        self.wait_for_stop(1)
        print('DetectionProcessor: %fs' % (np.mean(time_rec)))
        self.stopped = True


class ImageCropper(PoseStage):
    '''Crop persons from original images'''
    def __init__(self, det_processor, queue_size=50):
        super(ImageCropper, self).__init__(self.process, queue_size, prev=det_processor)
        self.data_len = len(det_processor)
        self.input_size = int(opt.inp_dim)

    def __len__(self):
        return self.data_len

    def process(self):
        time_rec = []
        for _ in range(self.data_len):
            if self.prev.stopped:
                break

            try:
                img, boxes, scores, img_name = self.prev.next(timeout=1)
            except Exception:
                continue

            tic = time.time()

            if boxes is None:
                self.wait_for_queue(0.1)
                self.Q.put((None, img, None, None, None, None, img_name))
                continue

            # crop person poses
            tensors, pt1, pt2 = self.crop_fn(img, boxes)

            # put into queue
            self.wait_for_queue(0.1)
            self.Q.put((tensors, img, boxes, scores, pt1, pt2, img_name))

            toc = time.time()
            time_rec.append(toc - tic)

        self.wait_for_stop(1)
        print('ImageCropper: %fs' % (np.mean(time_rec)))
        self.stopped = True

    def crop_fn(self, img, boxes):
        '''
        Crop persons based on given boxes
        INPUT:
            img: mx.nd, original image
            boxes: mx.nd, image size after resize
        OUTPUT:
            tensors: mx.nd, input tensor for pose estimation
            pt1: mx.nd, coordinates of left upper box corners
            pt2: mx.nd, coordinates of right bottom box corners
        '''
        mean = (0.485, 0.456, 0.406)
        std = (1.0, 1.0, 1.0)
        img_width, img_height = img.shape[1], img.shape[0]

        tensors = mx.nd.zeros([boxes.shape[0], 3, opt.inputResH, opt.inputResW])
        pt1 = mx.nd.zeros([boxes.shape[0], 2])
        pt2 = mx.nd.zeros([boxes.shape[0], 2])

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        img = img.transpose(axes=[1, 2, 0])

        for i, box in enumerate(boxes.asnumpy()):
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            if box_width > 100:
                scale_rate = 0.2
            else:
                scale_rate = 0.3

            # crop image
            left = int(max(0, box[0] - box_width * scale_rate / 2))
            up = int(max(0, box[1] - box_height * scale_rate / 2))
            right = int(min(img_width - 1,
                            max(left + 5, box[2] + box_width * scale_rate / 2)))
            bottom = int(min(img_height - 1,
                             max(up + 5, box[3] + box_height * scale_rate / 2)))
            crop_width = right - left
            crop_height = bottom - up
            cropped_img = mx.image.fixed_crop(img, left, up, crop_width, crop_height)

            # resize image
            resize_factor = min(opt.inputResW / crop_width, opt.inputResH / crop_height)
            new_width = int(crop_width * resize_factor)
            new_height = int(crop_height * resize_factor)
            tensor = mx.image.imresize(cropped_img, new_width, new_height)
            tensor = tensor.transpose(axes=[2, 0, 1])
            tensor = tensor.reshape(1, 3, new_height, new_width)

            # pad tensor
            pad_h = opt.inputResH - new_height
            pad_w = opt.inputResW - new_width
            pad_shape = (0, 0, 0, 0, pad_h // 2, (pad_h + 1) // 2, pad_w // 2, (pad_w + 1) // 2)
            tensor = mx.nd.pad(tensor, mode='constant',
                               constant_value=0.5, pad_width=pad_shape)
            tensors[i] = tensor.reshape(3, opt.inputResH, opt.inputResW)
            pt1[i] = (left, up)
            pt2[i] = (right, bottom)

        return tensors, pt1, pt2


class PoseEstimator(PoseStage):
    '''Estimate person poses and transform pose coordinates'''
    def __init__(self, img_cropper, batch_size=1, queue_size=1024):
        super(PoseEstimator, self).__init__(self.estimate_fn, queue_size, prev=img_cropper)
        self.data_len = len(img_cropper)
        self.batch_size = batch_size

        # model config
        print('Loading SPPE ...')
        self.net = FastPose_SE(ctx)
        self.net.load_parameters('sppe/params/duc_se.params')
        self.net.hybridize()
        self.net.collect_params().reset_ctx(ctx)

    def __len__(self):
        return self.data_len

    def estimate_fn(self):
        time_rec = []
        for _ in tqdm.tqdm(range(self.data_len)):
            if self.prev.stopped:
                break

            try:
                tensors, img, boxes, box_scores, pt1, pt2, img_name = self.prev.next(timeout=1)
            except Exception:
                continue

            tic = time.time()

            if tensors is None:
                self.wait_for_queue(0.1)
                self.Q.put((img, None, None, None, None, img_name))
                continue

            heatmaps = []
            num_poses = tensors.shape[0]
            num_batches = (num_poses + self.batch_size - 1) // self.batch_size

            for k in range(num_batches):
                # get batch tensor
                begin_idx = k * self.batch_size
                end_idx = min(begin_idx + self.batch_size, num_poses)
                tensor_batch = tensors[begin_idx:end_idx]
                # get prediction
                heatmap_batch = self.net(tensor_batch.copyto(ctx))
                heatmap_batch = heatmap_batch[:, :17, :, :]
                heatmaps.append(heatmap_batch.copyto(mx.cpu()))

            # coordinate transformation
            heatmaps = mx.nd.concatenate(heatmaps, axis=0)
            pose_hms, pose_coords, pose_scores = self.transform_fn(heatmaps, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

            # put into queue
            self.wait_for_queue(0.1)
            self.Q.put((img, boxes, box_scores, pose_coords, pose_scores, img_name))

            toc = time.time()
            time_rec.append(toc - tic)

        self.wait_for_stop(1)
        print('PoseEstimator: %fs' % (np.mean(time_rec)))
        self.stopped = True

    def transform_fn(self, hms, pt1, pt2, inp_h, inp_w, res_h, res_w):
        '''
        Transform pose heatmaps to coordinates
        INPUT:
            hms: mx.nd, pose heatmaps
            pt1: mx.nd, coordinates of left upper box corners
            pt2: mx.nd, coordinates of right bottom box corners
            inp_h: int, input tensor height
            inp_w: int, input tensot width
            res_h: int, output heatmap height
            res_w: int, output heatmap width
        OUTPUT:
            preds: mx.nd, pose coordinates in box frames
            preds_tf: mx.nd, pose coordinates in image frames
            maxval: mx.nd, pose scores
        '''
        pt1 = pt1.expand_dims(axis=1)
        pt2 = pt2.expand_dims(axis=1)

        # get keypoint coordinates
        idxs = mx.nd.argmax(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
        maxval = mx.nd.max(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
        preds = idxs.tile(reps=[1, 1, 2])
        preds[:, :, 0] %= hms.shape[3]
        preds[:, :, 1] /= hms.shape[3]

        # get pred masks
        pred_mask = (maxval > 0).tile(reps=[1, 1, 2])
        preds *= pred_mask

        # coordinate transformation
        box_size = pt2 - pt1
        len_h = mx.nd.maximum(box_size[:, :, 1:2], box_size[:, :, 0:1] * inp_h / inp_w)
        len_w = len_h * inp_w / inp_h
        canvas_size = mx.nd.concatenate([len_w, len_h], axis=2)
        offsets = pt1 - mx.nd.maximum(0, canvas_size / 2 - box_size / 2)
        preds_tf = preds * len_h / res_h + offsets

        return preds, preds_tf, maxval


class PoseProcessor(PoseStage):
    '''Pose NMS'''
    def __init__(self, pose_estimator, queue_size=1024):
        super(PoseProcessor, self).__init__(self.process, queue_size, prev=pose_estimator)
        self.data_len = len(pose_estimator)

    def __len__(self):
        return self.data_len

    def process(self):
        time_rec = []
        # im_names_desc = tqdm.tqdm(range(self.data_len))
        im_names_desc = range(self.data_len)
        for _ in im_names_desc:
            if self.prev.stopped:
                break

            try:
                img, boxes, box_scores, pose_coords, pose_scores, img_name = self.prev.next(timeout=1)
            except Exception:
                continue

            tic = time.time()

            if boxes is None:
                self.wait_for_queue(0.1)
                self.Q.put((img, None, None, None, img_name))
                continue

            # pose nms
            final_result, boxes, box_scores = pose_nms(boxes.asnumpy(),
                                                       box_scores.asnumpy(),
                                                       pose_coords.asnumpy(), pose_scores.asnumpy())

            toc = time.time()
            time_rec.append(toc - tic)

            # put into queue
            self.wait_for_queue(0.1)
            if opt.save_img:
                gcv.utils.viz.plot_bbox(img, boxes, box_scores, thresh=0.1)
                plt.xlim([0, img.shape[1] - 1])
                plt.ylim([0, img.shape[0] - 1])
                plt.gca().invert_yaxis()
                for result in final_result:
                    pts = result['keypoints']
                    mask = (result['kp_score'][:, 0] > 0.1)
                    plt.scatter(pts[:, 0][mask], pts[:, 1][mask], s=20)
                plt.axis('off')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.savefig(os.path.join('examples/res', img_name.split('/')[-1]))

            self.Q.put((img, final_result, boxes, box_scores, img_name))

        # self.wait_for_stop(1)
        while not self.prev.stopped:
            time.sleep(1)
        print('PoseProcessor: %fs' % (np.mean(time_rec)))
        self.stopped = True
