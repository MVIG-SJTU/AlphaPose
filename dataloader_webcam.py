import os
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from opt import opt
from yolo.preprocess import prep_image, prep_frame, inp_to_image
from pPose_nms import pose_nms, write_json
from SPPE.src.utils.eval import getPrediction
from yolo.util import write_results, dynamic_write_results
from yolo.darknet import Darknet
from tqdm import tqdm
import cv2
import json
import numpy as np
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue

if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame


class WebcamLoader:
    def __init__(self, webcam, batchSize=1, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(int(webcam))
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.batchSize = batchSize
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        i = 0
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                img = []
                orig_img = []
                im_name = []
                im_dim_list = []
                for k in range(self.batchSize):
                    (grabbed, frame) = self.stream.read()
                    # if the `grabbed` boolean is `False`, then we have
                    # reached the end of the video file
                    if not grabbed:
                        self.stop()
                        return
                    inp_dim = int(opt.inp_dim)
                    img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
                
                    img.append(img_k)
                    orig_img.append(orig_img_k)
                    im_name.append(str(i)+'.jpg')
                    im_dim_list.append(im_dim_list_k)

                with torch.no_grad():
                    # Human Detection
                    img = torch.cat(img)
                    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

                    self.Q.put((img, orig_img, im_name, im_dim_list))
                    i = i+1

            else:
                with self.Q.mutex:
                    self.Q.queue.clear()
    def videoinfo(self):
        # indicate the video info
        fourcc=int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps=self.stream.get(cv2.CAP_PROP_FPS)
        frameSize=(int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc,fps,frameSize)

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue size
        return self.Q.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class DetectionLoader:
    def __init__(self, dataloder, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stopped = False
        self.dataloder = dataloder
        self.batchSize = batchSize
        # initialize the queue used to store frames read from
        # the video file
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping the whole dataset
        while True:
            img, orig_img, im_name, im_dim_list = self.dataloder.getitem()
            with self.dataloder.Q.mutex:
                self.dataloder.Q.queue.clear()
            with torch.no_grad():
                # Human Detection
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                    opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_img)):
                        if self.Q.full():
                            time.sleep(2)
                        self.Q.put((orig_img[k], im_name[k], None, None, None, None, None))
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                
                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

            for k in range(len(orig_img)):
                boxes_k = boxes[dets[:,0]==k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img[k], im_name[k], None, None, None, None, None))
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes_k.size(0), 2)
                pt2 = torch.zeros(boxes_k.size(0), 2)
                if self.Q.full():
                    time.sleep(2)
                self.Q.put((orig_img[k], im_name[k], boxes_k, scores[dets[:,0]==k], inps, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DetectionProcessor:
    def __init__(self, detectionLoader, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.detectionLoader = detectionLoader
        self.stopped = False

        # initialize the queue used to store data
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping the whole dataset
        while True:
            
            with torch.no_grad():
                (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()
                with self.detectionLoader.Q.mutex:
                    self.detectionLoader.Q.queue.clear()
                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, im_name, boxes, scores, None, None))
                    continue
                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class WebcamDetectionLoader:
    def __init__(self, webcam = 0, batchSize=1, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stream = cv2.VideoCapture(int(webcam))
        assert self.stream.isOpened(), 'Cannot open webcam'
        self.stopped = False
        self.batchSize = batchSize

        # initialize the queue used to store frames read from
        # the video file
        self.Q = LifoQueue(maxsize=queueSize)


    def len(self):
        return self.Q.qsize()

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping
        while True:
            img = []
            inp = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(self.batchSize):
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    continue
                # process and add the frame to the queue
                inp_dim = int(opt.inp_dim)
                img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
                inp_k = im_to_torch(orig_img_k)

                img.append(img_k)
                inp.append(inp_k)
                orig_img.append(orig_img_k)
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                ht = inp[0].size(1)
                wd = inp[0].size(2)
                # Human Detection
                img = Variable(torch.cat(img)).cuda()
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
                im_dim_list = im_dim_list.cuda()

                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                    opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(inp)):
                        if self.Q.full():
                            with self.Q.mutex:
                                self.Q.queue.clear()
                        self.Q.put((inp[k], orig_img[k], None, None))
                    continue

                im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5].cpu()
                scores = dets[:, 5:6].cpu()

            for k in range(len(inp)):
                if self.Q.full():
                    with self.Q.mutex:
                        self.Q.queue.clear()
                self.Q.put((inp[k], orig_img[k], boxes[dets[:,0]==k], scores[dets[:,0]==k]))

    def videoinfo(self):
        # indicate the video info
        fourcc=int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps=self.stream.get(cv2.CAP_PROP_FPS)
        frameSize=(int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc,fps,frameSize)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True



class DataWriter:
    def __init__(self, save_video=False,
                savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480),
                queueSize=1024):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (boxes, scores, hm_data, pt1, pt2, orig_img, im_name) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)
                if boxes is None:
                    if opt.save_img or opt.save_video or opt.vis:
                        img = orig_img
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
                else:
                    # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                    
                    preds_hm, preds_img, preds_scores = getPrediction(
                        hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

                    result = pose_nms(boxes, scores, preds_img, preds_scores)
                    result = {
                        'imgname': im_name,
                        'result': result
                    }
                    self.final_result.append(result)
                    if opt.save_img or opt.save_video or opt.vis:
                        img = vis_frame(orig_img, result)
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()

class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        inps[i] = cropBox(tmp_img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
