import math
import time

import cv2
import matplotlib

matplotlib.use('agg')
import logging

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil_img
import torch

logging.getLogger('matplotlib.font_manager').disabled = True

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color


def get_smpl_color(idx):
    color_pool = [
        [int(0.65098039 * 255), int(0.74117647 * 255), int(0.85882353 * 255)],
        [0.9 * 255, 0.7 * 255, 0.7 * 255],
        [120, 198, 121],
        [0.74117647 * 255, 0.65098039 * 255, 0.85882353 * 255],
        [0.7 * 255, 0.9 * 255, 0.7 * 255],
        [198, 120, 121],
        CYAN, WHITE]
    color = color_pool[idx % 8]

    return color


def vis_frame_fast(frame, im_res, opt, vis_thres, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    kp_num = 17
    if len(im_res['result']) > 0:
        kp_num = len(im_res['result'][0]['keypoints'])
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    elif kp_num == 136:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
            (26, 27),(27, 28),(28, 29),(29, 30),(30, 31),(31, 32),(32, 33),(33, 34),(34, 35),(35, 36),(36, 37),(37, 38),#Face
            (38, 39),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(46, 47),(48, 49),(49, 50),(50, 51),(51, 52),#Face
            (53, 54),(54, 55),(55, 56),(57, 58),(58, 59),(59, 60),(60, 61),(62, 63),(63, 64),(64, 65),(65, 66),(66, 67),#Face
            (68, 69),(69, 70),(70, 71),(71, 72),(72, 73),(74, 75),(75, 76),(76, 77),(77, 78),(78, 79),(79, 80),(80, 81),#Face
            (81, 82),(82, 83),(83, 84),(84, 85),(85, 86),(86, 87),(87, 88),(88, 89),(89, 90),(90, 91),(91, 92),(92, 93),#Face
            (94,95),(95,96),(96,97),(97,98),(94,99),(99,100),(100,101),(101,102),(94,103),(103,104),(104,105),#LeftHand
            (105,106),(94,107),(107,108),(108,109),(109,110),(94,111),(111,112),(112,113),(113,114),#LeftHand
            (115,116),(116,117),(117,118),(118,119),(115,120),(120,121),(121,122),(122,123),(115,124),(124,125),#RightHand
            (125,126),(126,127),(115,128),(128,129),(129,130),(130,131),(115,132),(132,133),(133,134),(134,135)#RightHand
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 133:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (11, 13), (12, 14), (13, 15), (14, 16),
            (18, 19), (21, 22), (20, 22), (17, 19), (15, 19), (16, 22), 
            (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), 
            (35, 36), (36, 37), (37, 38), (38, 39), (40, 41), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47), (47, 48), (48, 49), 
            (50, 51), (51, 52), (52, 53), (54, 55), (55, 56), (56, 57), (57, 58), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), 
            (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), 
            (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), 
            (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98), (98, 99), (91, 100), (100, 101), (101, 102), 
            (102, 103), (91, 104), (104, 105), (105, 106), (106, 107), (91, 108), (108, 109), (109, 110), (110, 111), (112, 113), 
            (113, 114), (114, 115), (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123), 
            (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131), (131, 132)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255)]
    elif kp_num == 68:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
            (26, 27), (27, 28), (28, 29), (29, 30), (26, 31), (31, 32), (32, 33), (33, 34), 
            (26, 35), (35, 36), (36, 37), (37, 38), (26, 39), (39, 40), (40, 41), (41, 42), 
            (26, 43), (43, 44), (44, 45), (45, 46), (47, 48), (48, 49), (49, 50), (50, 51), 
            (47, 52), (52, 53), (53, 54), (54, 55), (47, 56), (56, 57), (57, 58), (58, 59), 
            (47, 60), (60, 61), (61, 62), (62, 63), (47, 64), (64, 65), (65, 66), (66, 67)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 21:
        l_pair = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), 
            (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (21, 22), (22, 23),
            (23, 24), (24, 25), (21, 26), (26, 27), (27, 28), (28, 29), (21, 30), 
            (30, 31), (31, 32), (32, 33), (21, 34), (34, 35), (35, 36), (36, 37), 
            (21, 38), (38, 39), (39, 40), (40, 41)
        ]
        p_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255) ]
    
        line_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255) ]
    else:
        raise NotImplementedError
    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        if kp_num == 17:
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            vis_thres.append(vis_thres[-1])
        if opt.pose_track or opt.tracking:
            while isinstance(human['idx'], list):
                human['idx'].sort()
                human['idx'] = human['idx'][0]
            color = get_color_fast(int(abs(human['idx'])))
        else:
            color = BLUE

        # Draw bboxes
        if opt.showbox:
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]#xmin,xmax,ymin,ymax
            else:
                from trackers.PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)
            
            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
            if opt.tracking:
                cv2.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres[n]:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            if n < len(p_color):
                if opt.tracking:
                    cv2.circle(img, (cor_x, cor_y), 3, color, -1)
                else:
                    cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
            else:
                cv2.circle(img, (cor_x, cor_y), 1, (255,255,255), 2)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    if opt.tracking:
                        cv2.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                    else:
                        cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                else:
                    cv2.line(img, start_xy, end_xy, (255,255,255), 1)

    return img


def vis_frame(frame, im_res, opt, vis_thres, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    kp_num = 17
    if len(im_res['result']) > 0:
        kp_num = len(im_res['result'][0]['keypoints'])

    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]

            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
            line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    elif kp_num == 136:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
            (26, 27),(27, 28),(28, 29),(29, 30),(30, 31),(31, 32),(32, 33),(33, 34),(34, 35),(35, 36),(36, 37),(37, 38),#Face
            (38, 39),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(46, 47),(48, 49),(49, 50),(50, 51),(51, 52),#Face
            (53, 54),(54, 55),(55, 56),(57, 58),(58, 59),(59, 60),(60, 61),(62, 63),(63, 64),(64, 65),(65, 66),(66, 67),#Face
            (68, 69),(69, 70),(70, 71),(71, 72),(72, 73),(74, 75),(75, 76),(76, 77),(77, 78),(78, 79),(79, 80),(80, 81),#Face
            (81, 82),(82, 83),(83, 84),(84, 85),(85, 86),(86, 87),(87, 88),(88, 89),(89, 90),(90, 91),(91, 92),(92, 93),#Face
            (94,95),(95,96),(96,97),(97,98),(94,99),(99,100),(100,101),(101,102),(94,103),(103,104),(104,105),#LeftHand
            (105,106),(94,107),(107,108),(108,109),(109,110),(94,111),(111,112),(112,113),(113,114),#LeftHand
            (115,116),(116,117),(117,118),(118,119),(115,120),(120,121),(121,122),(122,123),(115,124),(124,125),#RightHand
            (125,126),(126,127),(115,128),(128,129),(129,130),(130,131),(115,132),(132,133),(133,134),(134,135)#RightHand
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 133:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (11, 13), (12, 14), (13, 15), (14, 16),
            (18, 19), (21, 22), (20, 22), (17, 19), (15, 19), (16, 22), 
            (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), 
            (35, 36), (36, 37), (37, 38), (38, 39), (40, 41), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47), (47, 48), (48, 49), 
            (50, 51), (51, 52), (52, 53), (54, 55), (55, 56), (56, 57), (57, 58), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), 
            (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), 
            (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), 
            (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98), (98, 99), (91, 100), (100, 101), (101, 102), 
            (102, 103), (91, 104), (104, 105), (105, 106), (106, 107), (91, 108), (108, 109), (109, 110), (110, 111), (112, 113), 
            (113, 114), (114, 115), (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123), 
            (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131), (131, 132)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255)]
    elif kp_num == 68:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
            (26, 27), (27, 28), (28, 29), (29, 30), (26, 31), (31, 32), (32, 33), (33, 34), 
            (26, 35), (35, 36), (36, 37), (37, 38), (26, 39), (39, 40), (40, 41), (41, 42), 
            (26, 43), (43, 44), (44, 45), (45, 46), (47, 48), (48, 49), (49, 50), (50, 51), 
            (47, 52), (52, 53), (53, 54), (54, 55), (47, 56), (56, 57), (57, 58), (58, 59), 
            (47, 60), (60, 61), (61, 62), (62, 63), (47, 64), (64, 65), (65, 66), (66, 67)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 21:
        l_pair = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), 
            (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (21, 22), (22, 23),
            (23, 24), (24, 25), (21, 26), (26, 27), (27, 28), (28, 29), (21, 30), 
            (30, 31), (31, 32), (32, 33), (21, 34), (34, 35), (35, 36), (36, 37), 
            (21, 38), (38, 39), (39, 40), (40, 41)
        ]
        p_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255) ]
    
        line_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                   (255, 255, 255) ]
    else:
        raise NotImplementedError
    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        if kp_num == 17:
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            vis_thres.append(vis_thres[-1])
        if opt.tracking:
            while isinstance(human['idx'], list):
                human['idx'].sort()
                human['idx'] = human['idx'][0]
            color = get_color_fast(int(abs(human['idx'])))
        else:
            color = BLUE

        # Draw bboxes
        if opt.showbox:
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]#xmin,xmax,ymin,ymax
            else:
                from trackers.PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)
            # color = get_color_fast(int(abs(human['idx'][0][0])))
            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]),int(bbox[3])), color, 1)
            if opt.tracking:
                cv2.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)

        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres[n]:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            if n < len(p_color):
                if opt.tracking:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, color, -1)
                else:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
            else:
                cv2.circle(bg, (int(cor_x), int(cor_y)), 1, (255,255,255), 2)
            # Now create a mask of logo and create its inverse mask also
            if n < len(p_color):
                transparency = float(max(0, min(1, kp_scores[n])))
            else:
                transparency = float(max(0, min(1, kp_scores[n]*2)))
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):
                    if opt.tracking:
                        cv2.fillConvexPoly(bg, polygon, color)
                    else:
                        cv2.fillConvexPoly(bg, polygon, line_color[i])
                else:
                    cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
                if n < len(p_color):
                    transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                else:
                    transparency = float(max(0, min(1, (kp_scores[start_p] + kp_scores[end_p]))))

                #transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    return img


def vis_frame_smpl(frame, im_res, smpl_output, opt, vis_thres):
    '''
    frame: frame image
    im_res: result dict
    smpl_output: predictions

    return rendered image
    '''
    from .render_pytorch3d import render_mesh
    
    img = frame.copy()
    height, width = img.shape[:2]
    img_size = (height, width)
    focal = np.array([1000, 1000])

    all_transl = smpl_output['transl'].detach()
    vertices = smpl_output['pred_vertices'].detach()
    smpl_faces = smpl_output['smpl_faces']
    # all_theta = pose_output.pred_theta_mats.detach().cpu().numpy()

    for n_human, human in enumerate(im_res['result']):
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        score = human['bbox_score']
        if score < 0.3:
            continue
        # bbox = human['box']
        # x1y1wh
        bbox = human['crop_box']
        bbox_w = bbox[2]
        # x1x2y1y2
        bbox = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]#xmin,xmax,ymin,ymax
        if opt.pose_track or opt.tracking:
            while isinstance(human['idx'], list):
                human['idx'].sort()
                human['idx'] = human['idx'][0]
            color = get_smpl_color(int(abs(human['idx'])))
        else:
            color = [int(0.65098039 * 255), int(0.74117647 * 255), int(0.85882353 * 255)]

        # Draw bboxes
        if opt.showbox:
            if 'crop_box' not in human.keys():
                from trackers.PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)

            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
            if opt.tracking:
                cv2.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)

        # Draw SMPL
        # princpt = [(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2]

        # renderer = SMPLRenderer(img_size=img_size, focal=focal,
        #                         princpt=princpt)
        transl = all_transl[[n_human]]
        # transl[2] = transl[2] * 256 / (bbox[1] - bbox[0])

        vert = vertices[[n_human]]
        # print(all_theta[n_human])

        # img = vis_smpl_3d(
        #     vert, img, cam_root=transl,
        #     f=focal, c=princpt, renderer=renderer, color=[c / 255 for c in color])
        img = vis_smpl_3d(
            vert, transl, img, bbox_w, smpl_faces, render_mesh, color=color
        )

    return img



def vis_smpl_3d(verts_batch, transl_batch, image, bbox_w, smpl_faces, render_mesh, color=None):
    focal = 1000.0

    focal = focal / 256 * bbox_w

    color_batch = render_mesh(
        vertices=verts_batch, faces=smpl_faces,
        translation=transl_batch,
        focal_length=focal, height=image.shape[0], width=image.shape[1],
        color=color)

    valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
    image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
    image_vis_batch = (image_vis_batch * 255).cpu().numpy()

    color = image_vis_batch[0]
    valid_mask = valid_mask_batch[0].cpu().numpy()
    input_img = image
    alpha = 0.9
    image_vis = alpha * color[:, :, :3] * valid_mask + (
        1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

    image_vis = image_vis.astype(np.uint8)
    # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    return image_vis


def vis_frame_skeleton(frame, im_res, smpl_output, opt, vis_thres):
    '''
    frame: frame image
    im_res: result dict
    smpl_output: predictions

    return rendered image
    '''
    img = frame.copy()
    height, width = img.shape[:2]
    focal = np.array([1000, 1000])

    all_transl = smpl_output['transl'].detach().cpu().numpy()

    l_pair = [(15, 12), (12, 9), 
        (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), 
        (9, 6), (6, 3), (3, 0),
        (0, 1), (1, 4), (4, 7), (7, 10),
        (0, 2), (2, 5), (5, 8), (8, 11)
    ]

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(l_pair) + 2)]
    colors = [np.array((c[0], c[1], c[2])) for c in colors]

    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax = fig.add_subplot(111, projection="3d", autoscale_on=False)

    for n_human, human in enumerate(im_res['result']):
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        xyz_preds = human['pred_xyz_jts']
        score = human['bbox_score']
        if score < 0.3:
            continue
        # x1y1wh
        bbox = human['crop_box']
        # x1x2y1y2
        bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]] # xmin,xmax,ymin,ymax

        if opt.pose_track or opt.tracking:
            while isinstance(human['idx'], list):
                human['idx'].sort()
                human['idx'] = human['idx'][0]
            color = get_smpl_color(int(abs(human['idx'])))
        else:
            color = BLUE
        
        # Draw bboxes
        if opt.showbox:
            if 'crop_box' not in human.keys():
                from trackers.PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)

            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
            if opt.tracking:
                cv2.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)

        # Draw keypoints
        for n in range(len(l_pair)):
            visible_1 = kp_scores[l_pair[n][0], 0] > vis_thres[l_pair[n][0]]
            visible_2 = kp_scores[l_pair[n][1], 0] > vis_thres[l_pair[n][1]]
            cor_x_1, cor_y_1 = int(kp_preds[l_pair[n][0], 0]), int(
                kp_preds[l_pair[n][0], 1]
            )
            cor_x_2, cor_y_2 = int(kp_preds[l_pair[n][1], 0]), int(
                kp_preds[l_pair[n][1], 1]
            )
            if opt.tracking:
                if visible_1:
                    cv2.circle(img, (cor_x_1, cor_y_1), 3, color, -1)
                if visible_2:
                    cv2.circle(img, (cor_x_2, cor_y_2), 3, color, -1)
                if visible_1 and visible_2:
                    cv2.line(
                        img,
                        (cor_x_1, cor_y_1),
                        (cor_x_2, cor_y_2),
                        color,
                        2 * int(kp_scores[l_pair[n][0]] + kp_scores[l_pair[n][1]]) + 1,
                    )
            else:
                if visible_1:
                    cv2.circle(
                        img,
                        (cor_x_1, cor_y_1),
                        3,
                        (
                            int(colors[n][0] * 255),
                            int(colors[n][1] * 255),
                            int(colors[n][2] * 255),
                        ),
                        -1,
                    )
                if visible_2:
                    cv2.circle(
                        img,
                        (cor_x_2, cor_y_2),
                        3,
                        (
                            int(colors[n][0] * 255),
                            int(colors[n][1] * 255),
                            int(colors[n][2] * 255),
                        ),
                        -1,
                    )
                if visible_1 and visible_2:
                    cv2.line(
                        img,
                        (cor_x_1, cor_y_1),
                        (cor_x_2, cor_y_2),
                        (
                            int(colors[n][0] * 255),
                            int(colors[n][1] * 255),
                            int(colors[n][2] * 255),
                        ),
                        2 * int(kp_scores[l_pair[n][0]] + kp_scores[l_pair[n][1]]) + 1,
                    )

        # Draw 3d skeleton
        transl = all_transl[n_human].squeeze()
        transl[0] = transl[0] + ((bbox[0] + bbox[1]) / 2) * transl[2] / focal[0]
        transl[1] = transl[1] + ((bbox[2] + bbox[3]) / 2) * transl[2] / focal[1]
        transl[2] = transl[2] * 256 / (bbox[1] - bbox[0])

        xyz_preds = xyz_preds + transl
        xyz_preds = xyz_preds.numpy()

        for l in range(len(l_pair)):
            i1 = l_pair[l][0]
            i2 = l_pair[l][1]
            x = np.array([xyz_preds[i1, 0], xyz_preds[i2, 0]])
            y = np.array([xyz_preds[i1, 1], xyz_preds[i2, 1]])
            z = np.array([xyz_preds[i1, 2], xyz_preds[i2, 2]])

            if kp_scores[i1, 0] > vis_thres[i1] and kp_scores[i2, 0] > vis_thres[i2]:
                ax.plot(
                    x,
                    z,
                    -y,
                    color=colors[l] if not opt.tracking else np.array(color) / 255,
                    linewidth=2,
                )
            if kp_scores[i1, 0] > vis_thres[i1]:
                ax.scatter(
                    xyz_preds[i1, 0],
                    xyz_preds[i1, 2],
                    -xyz_preds[i1, 1],
                    color=colors[l] if not opt.tracking else np.array(color) / 255,
                    marker="o",
                )
            if kp_scores[i2, 0] > vis_thres[i2]:
                ax.scatter(
                    xyz_preds[i2, 0],
                    xyz_preds[i2, 2],
                    -xyz_preds[i2, 1],
                    color=colors[l] if not opt.tracking else np.array(color) / 255,
                    marker="o",
                )


    ax.set_xlim([-2, 6])
    ax.set_ylim([14, 20])
    ax.set_zlim([-4, 0])

    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    # ax.axes.zaxis.set_ticklabels([])

    ax.view_init(azim=-70, elev=15)

    # Convert plt to cv2
    skeleton_img = mplfig_to_npimage(fig)
    skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    cat_img = np.ones(
        (
            max(skeleton_img.shape[0], img.shape[0]),
            skeleton_img.shape[1] + img.shape[1],
            3,
        ), dtype=np.uint8
    ) * 255
    cat_img[
        (cat_img.shape[0] - img.shape[0]) // 2 : (cat_img.shape[0] - img.shape[0]) // 2
        + img.shape[0],
        : img.shape[1],
        :,
    ] = img
    cat_img[
        (cat_img.shape[0] - skeleton_img.shape[0])
        // 2 : (cat_img.shape[0] - skeleton_img.shape[0])
        // 2
        + skeleton_img.shape[0],
        img.shape[1] :,
        :,
    ] = skeleton_img

    return cat_img


def getTime(time1=0):
    if not time1:

        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval


def mplfig_to_npimage(fig):
    """
    Converts a matplotlib figure to a RGB frame after updating the canvas.
    Modified from https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/bindings.py
    """
    #  only the Agg backend now supports the tostring_rgb function
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # update/draw the elements

    # get the width and the height to resize the matrix
    l, b, w, h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    # exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.tostring_rgb()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

    b = fig.axes[0].get_window_extent()
    image = image[int(b.y0) : int(b.y1), int(b.x0) : int(b.x1), :]

    return image