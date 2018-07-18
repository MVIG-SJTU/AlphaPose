import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from opt import opt
from yolo.preprocess import prep_image, prep_frame, inp_to_image
from pPose_nms import pose_nms, write_json
from SPPE.src.utils.eval import getPrediction

import cv2
import json
import sys
import time
from threading import Thread
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue

if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame


class Image_loader(data.Dataset):
    def __init__(self, im_names, format='yolo'):
        super(Image_loader, self).__init__()
        self.img_dir = opt.inputpath
        self.imglist = im_names
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.format = format

    def getitem_ssd(self, index):
        im_name = self.imglist[index].rstrip('\n').rstrip('\r')
        im_name = os.path.join(self.img_dir, im_name)
        im = Image.open(im_name)
        inp = load_image(im_name)
        if im.mode == 'L':
            im = im.convert('RGB')

        ow = oh = 512
        im = im.resize((ow, oh))
        im = self.transform(im)
        return im, inp, im_name

    def getitem_yolo(self, index):
        inp_dim = int(opt.inp_dim)
        im_name = self.imglist[index].rstrip('\n').rstrip('\r')
        im_name = os.path.join(self.img_dir, im_name)
        im, orig_img, im_dim = prep_image(im_name, inp_dim)
        im_dim = torch.FloatTensor([im_dim]).repeat(1, 2)

        inp = load_image(im_name)
        return im, inp, orig_img, im_name, im_dim

    def __getitem__(self, index):
        if self.format == 'ssd':
            return self.getitem_ssd(index)
        elif self.format == 'yolo':
            return self.getitem_yolo(index)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.imglist)


class VideoLoader:
    def __init__(self, path, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        self.len = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def length(self):
        return self.len

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            time.sleep(0.02)
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # process and add the frame to the queue
                inp_dim = int(opt.inp_dim)
                img, orig_img, dim = prep_frame(frame, inp_dim)
                inp = im_to_torch(orig_img)
                im_dim_list = torch.FloatTensor([dim]).repeat(1, 2)

                self.Q.put((img, orig_img, inp, im_dim_list))

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
                if boxes is None:
                    if opt.save_img or opt.save_video:
                        #img = display_frame(orig_img, result, opt.outputpath)
                        img = orig_img
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
                    if opt.save_img or opt.save_video:
                        #img = display_frame(orig_img, result, opt.outputpath)
                        img = vis_frame(orig_img, result)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
            else:
                time.sleep(0.01)

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
        time.sleep(0.02)

    def results(self):
        # return final result
        return self.final_result


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


def crop_from_dets(img, boxes):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    inp = []
    pt1 = []
    pt2 = []
    for box in boxes:
        tmp_img = img.clone()
        tmp_img[0].add_(-0.406)
        tmp_img[1].add_(-0.457)
        tmp_img[2].add_(-0.480)

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

        tmp_inp = cropBox(tmp_img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
        inp.append(tmp_inp.unsqueeze(0))
        pt1.append(upLeft.unsqueeze(0))
        pt2.append(bottomRight.unsqueeze(0))

    inp = torch.cat(inp, 0)
    pt1 = torch.cat(pt1, 0)
    pt2 = torch.cat(pt2, 0)

    return inp, pt1, pt2
