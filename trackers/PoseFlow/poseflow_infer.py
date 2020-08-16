# -*- coding: utf-8 -*-
# @Author: Chao Xu
# @Email: xuchao.19962007@sjtu.edu.cn
# @Date:   2019-10-09 17:42:10
# @Last Modified by:   Chao Xu
# @Last Modified time: 2019-10-27 20:20:45

import os
import numpy as np

from .matching import orb_matching
from .utils import expand_bbox, stack_all_pids, best_matching_hungarian

def get_box(pose, img_height, img_width):

    pose = np.array(pose).reshape(-1,3)
    xmin = np.min(pose[:,0])
    xmax = np.max(pose[:,0])
    ymin = np.min(pose[:,1])
    ymax = np.max(pose[:,1])

    return expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height)

#The wrapper of PoseFlow algorithm to be embedded in alphapose inference
class PoseFlowWrapper():
    def __init__(self, link=100, drop=2.0, num=7,
                 mag=30, match=0.2, save_path='.tmp/poseflow', pool_size=5):
        # super parameters
        # 1. look-ahead LINK_LEN frames to find tracked human bbox
        # 2. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score
        # 3. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score(Non DeepMatching)
        # 4. drop low-score(<DROP) keypoints
        # 5. pick high-score(top NUM) keypoints when computing pose_IOU
        # 6. box width/height around keypoint for computing pose IoU
        # 7. match threshold in Hungarian Matching
        self.link_len = link
        self.weights = [1,2,1,2,0,0] 
        self.weights_fff = [0,1,0,1,0,0]
        self.drop = drop
        self.num = num
        self.mag = mag
        self.match_thres = match
        self.notrack = {}
        self.track = {}
        self.save_path = save_path
        self.save_match_path = os.path.join(save_path,'matching')
        self.pool_size = pool_size

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        #init local variables
        self.max_pid_id = 0
        self.prev_img = None
        print("Start pose tracking...\n")

    def convert_results_to_no_track(self, alphapose_results):
        # INPUT:
        #   alphapose_results: the results of pose detection given by pose_nms,
        #   not the final version for saving. Data array's format is torch.FloatTensor.
        #   format: {"imgname": str, "result": [{'keypoints': [17,2], 'kp_score': [17,], 'proposal_score': float},...]}
        # OUTPUT:
        #   notrack: data array's format is list.
        #   format: {"(str)$imgid": [{'keypoints': [17*3], 'scores': float},...]}
        imgname = os.path.basename(alphapose_results["imgname"])
        alphapose_results = alphapose_results["result"]
        notrack = {}
        notrack[imgname] = []
        for human in alphapose_results:
            keypoints = []
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            notrack[imgname].append({'keypoints': keypoints, 'scores': pro_scores})
        return notrack

    def convert_notrack_to_track(self, notrack, img_height, img_width):
        # INPUT:
        #   notrack: data array's format is list.
        #   - format: {"(str)$imgid": [{'keypoints': [17*3], 'scores': float},...]}
        #   img_height: int
        #   img_width: int
        # OUTPUT:
        #   track: tracked human poses
        #   - format: {'num_boxes': int, '$1-indexed human id': {'box_score': float, 'box_pos': [4], 
        #              'box_pose_pos': [17,2], 'box_pose_score': [17,1]}, ...}
        track = {}
        for img_name in sorted(notrack.keys()):
            track[img_name] = {'num_boxes':len(notrack[img_name])}
            for bid in range(len(notrack[img_name])):
                track[img_name][bid+1] = {}
                track[img_name][bid+1]['box_score'] = notrack[img_name][bid]['scores']
                track[img_name][bid+1]['box_pos'] = get_box(notrack[img_name][bid]['keypoints'], img_height, img_width)
                track[img_name][bid+1]['box_pose_pos'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,0:2]
                track[img_name][bid+1]['box_pose_score'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,-1]
        return track

    def step(self, img, alphapose_results):
        frame_name = os.path.basename(alphapose_results["imgname"])
        frame_id = frame_name.split(".")[0]
        #load track information
        _notrack = self.convert_results_to_no_track(alphapose_results)
        self.notrack.update(_notrack)
        img_height, img_width, _ = img.shape
        _track = self.convert_notrack_to_track(_notrack, img_height, img_width)
        self.track.update(_track)

        #track
        # init tracking info of the first frame in one video
        if len(self.track.keys()) == 1:
            for pid in range(1, self.track[frame_name]['num_boxes']+1):
                self.track[frame_name][pid]['new_pid'] = pid
                self.track[frame_name][pid]['match_score'] = 0
            #make directory to store matching files
            if not os.path.exists(self.save_match_path):
                os.mkdir(self.save_match_path)
            self.prev_img = img.copy()
            return self.final_result_by_name(frame_name)

        frame_id_list = sorted([(int(os.path.splitext(i)[0]), os.path.splitext(i)[1]) for i in self.track.keys()])
        frame_list = [ "".join([str(i[0]), i[1]]) for i in frame_id_list]
        prev_frame_name = frame_list[-2]
        prev_frame_id = prev_frame_name.split(".")[0]
        frame_new_pids = []

        self.max_pid_id = max(self.max_pid_id, self.track[prev_frame_name]['num_boxes'])
        cor_file = os.path.join(self.save_match_path, "".join([prev_frame_id, '_', frame_id, '_orb.txt']))
        orb_matching(self.prev_img, img, self.save_match_path, prev_frame_id, frame_id)
        all_cors = np.loadtxt(cor_file)

        if self.track[frame_name]['num_boxes'] == 0:
            self.track[frame_name] = copy.deepcopy(self.track[prev_frame_name])
            self.prev_img = img.copy()
            return self.final_result_by_name(frame_name)

        cur_all_pids, cur_all_pids_fff = stack_all_pids(self.track, frame_list, len(frame_list)-2, self.max_pid_id, self.link_len)
        match_indexes, match_scores = best_matching_hungarian(
            all_cors, cur_all_pids, cur_all_pids_fff, self.track[frame_name], self.weights, self.weights_fff, self.num, self.mag, pool_size=self.pool_size)

        for pid1, pid2 in match_indexes:
            if match_scores[pid1][pid2] > self.match_thres:
                self.track[frame_name][pid2+1]['new_pid'] = cur_all_pids[pid1]['new_pid']
                self.max_pid_id = max(self.max_pid_id, self.track[frame_name][pid2+1]['new_pid'])
                self.track[frame_name][pid2+1]['match_score'] = match_scores[pid1][pid2]

        # add the untracked new person
        for next_pid in range(1, self.track[frame_name]['num_boxes'] + 1):
            if 'new_pid' not in self.track[frame_name][next_pid]:
                self.max_pid_id += 1
                self.track[frame_name][next_pid]['new_pid'] = self.max_pid_id
                self.track[frame_name][next_pid]['match_score'] = 0

        self.prev_img = img.copy()
        return self.final_result_by_name(frame_name)

    @property
    def num_persons(self):
        # calculate number of people
        num_persons = 0
        frame_list = sorted(list(self.track.keys()))
        for fid, frame_name in enumerate(frame_list):
            for pid in range(1, self.track[frame_name]['num_boxes']+1):
                num_persons = max(num_persons, self.track[frame_name][pid]['new_pid'])
        return num_persons

    @property
    def final_results(self):
        # export tracking result into notrack json data
        frame_list = sorted(list(self.track.keys()))
        for fid, frame_name in enumerate(frame_list):
            for pid in range(self.track[frame_name]['num_boxes']):
                self.notrack[frame_name][pid]['idx'] = self.track[frame_name][pid+1]['new_pid']
        return self.notrack

    def final_result_by_name(self, frame_name):
        # export tracking result into notrack json data by frame name
        for pid in range(self.track[frame_name]['num_boxes']):
            self.notrack[frame_name][pid]['idx'] = self.track[frame_name][pid+1]['new_pid']
        return self.notrack[frame_name]







        