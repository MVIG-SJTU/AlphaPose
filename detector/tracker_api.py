# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoshu Fang (fhaoshu@gmail.com)
# -----------------------------------------------------

"""API of yolo tracker"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from abc import ABC, abstractmethod
import platform

import torch
import numpy as np

from tracker.tracker.multitracker import STrack, joint_stracks, sub_stracks, remove_duplicate_stracks

from tracker.preprocess import prep_image, prep_frame
from tracker.utils.kalman_filter import KalmanFilter
from tracker.utils.utils import non_max_suppression, scale_coords
from tracker.utils.log import logger
from tracker.tracker import matching
from tracker.tracker.basetrack import BaseTrack, TrackState
from tracker.models import Darknet

from detector.apis import BaseDetector


class Tracker(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(Tracker, self).__init__()

        self.tracker_opt = opt
        self.model_cfg = cfg.get('CONFIG', 'detector/tracker/cfg/yolov3.cfg')
        self.model_weights = cfg.get('WEIGHTS', 'detector/tracker/data/jde.1088x608.uncertainty.pt')
        self.img_size = cfg.get('IMG_SIZE', (1088, 608))
        self.nms_thres = cfg.get('NMS_THRES', 0.6)
        self.confidence = cfg.get('CONFIDENCE', 0.05)
        self.max_time_lost = cfg.get('BUFFER_SIZE', 30) # buffer
        self.model = None
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.emb_dim = None
        

        self.kalman_filter = KalmanFilter()

    def load_model(self):
        print('Loading tracking model..')
        self.model = Darknet(self.model_cfg, self.img_size, nID=14455)
        # load_darknet_weights(self.model, args.weights)
        self.model.load_state_dict(torch.load(self.model_weights, map_location='cpu')['model'], strict=False)
        self.emb_dim = self.model.emb_dim

        if self.tracker_opt:
            if len(self.tracker_opt.gpus) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.tracker_opt.gpus).to(self.tracker_opt.device)
            else:
                self.model.to(self.tracker_opt.device)
        else:
            self.model.cuda()
        self.model.eval()
        print("Network successfully loaded")

        

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        if isinstance(img_source, str):
            img, orig_img, im_dim_list = prep_image(img_source, self.img_size)
        elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
            img, orig_img, im_dim_list = prep_frame(img_source, self.img_size)
        else:
            raise IOError('Unknown image source type: {}'.format(type(img_source)))

        return img

    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into object detection network and 
        collect bbox w.r.t original image size
        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size
        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results
        """
        args = self.tracker_opt
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()
        
        
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            imgs = imgs.to(args.device) if args else imgs.cuda()
            pred = self.model(imgs)

        if len(pred) > 0:
            dets = non_max_suppression(pred, self.confidence, self.nms_thres)

        output_stracks = []
        for image_i in range(len(imgs)):
            self.frame_id += 1
            if dets[image_i] is not None:
                det_i = scale_coords(self.img_size, dets[image_i], orig_dim_list[image_i])
                '''Detections'''
                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                              (tlbrs, f) in zip(det_i[:, :5], det_i[:, -self.emb_dim:])]
            else:
                detections = []


            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed = []
            tracked_stracks = []  # type: list[STrack]
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)

            ''' Step 2: First association, with embedding'''
            strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
            # Predict the current location with KF
            for strack in strack_pool:
                strack.predict()

            dists = matching.embedding_distance(strack_pool, detections)
            dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            ''' Step 3: Second association, with IOU'''
            detections = [detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked ]
            dists = matching.iou_distance(r_tracked_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
            
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            """ Step 4: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.confidence:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

            """ Step 5: Update state"""
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)

            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
            self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
            # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
            self.removed_stracks.extend(removed_stracks)
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

            if self.tracker_opt.debug:
                logger.debug('===========Frame {}=========='.format(self.frame_id))
                logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
                logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
                logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
                logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

            # Add tracks to outputs
            for t in self.tracked_stracks:
                tlwh = t.tlwh
                tid = t.track_id
                tlbr = t.tlbr
                ts = t.score
                if tlwh[2] * tlwh[3] > self.tracker_opt.min_box_area:
                    res = torch.tensor([image_i, tlbr[0], tlbr[1], tlbr[2], tlbr[3], ts, tid])
                output_stracks.append(res)

        if len(output_stracks) == 0:
            return 0
            
        return torch.stack(output_stracks)

    def detect_one_img(self, img_name):
        pass