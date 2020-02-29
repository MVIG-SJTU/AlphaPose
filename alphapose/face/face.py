import os
import time

import cv2
import numpy as np


from alphapose.face.centerface import CenterFace
from alphapose.face.prnet import PRN
from alphapose.face.utils.cv_plot import plot_kpt, plot_pose_box, plot_vertices
from alphapose.face.utils.render_app import get_visibility, get_uv_mask, get_depth_image

current_path = os.path.dirname(__file__)


#useless path, enter anything
face_model_path = '../face/models/onnx/centerface.onnx'
face_engine = CenterFace(model_path=face_model_path, landmarks=True)

#useless path, enter anything
face_3d_model_path = '../face/models/prnet.pth'

face_3d_model = PRN(face_3d_model_path, '../face')

colors = [tuple(np.random.choice(np.arange(256).astype(np.int32), size=3)) for i in range(100)]


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('nose'), keypoints.index('left_eye')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
    ]
    return kp_lines


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = [
        'nose',  # 1
        'left_eye',  # 2
        'right_eye',  # 3
        'left_ear',  # 4
        'right_ear',  # 5
        'left_shoulder',  # 6
        'right_shoulder',  # 7
        'left_elbow',  # 8
        'right_elbow',  # 9
        'left_wrist',  # 10
        'right_wrist',  # 11
        'left_hip',  # 12
        'right_hip',  # 13
        'left_knee',  # 14
        'right_knee',  # 15
        'left_ankle',  # 16
        'right_ankle',  # 17
    ]

    return keypoints


_kp_connections = kp_connections(get_keypoints())


def face_process(result, rgb_img, orig_img, boxes, scores, ids, preds_img, preds_scores):
    boxes = boxes.numpy()

    i = 0
    face_engine.transform(orig_img.shape[0], orig_img.shape[1])
    face_dets, lms = face_engine(orig_img, threshold=0.35)

    bbox_xywh = []
    cls_conf = []

    for person in result:

        keypoints = person['keypoints']
      
        keypoints = keypoints.numpy()

        bbox = boxes[i]
        color = colors[i]

        body_prob = scores.numpy()

        body_bbox = np.array(bbox[:4], dtype=np.int32)
        w = body_bbox[2] - body_bbox[0]
        h = body_bbox[3] - body_bbox[1]
        bbox_xywh.append([body_bbox[0], body_bbox[1], w, h])
        cls_conf.append(body_prob)

        center_of_the_face = np.mean(keypoints[:7, :], axis=0)

        image = orig_img

        if len(face_dets) != 0:
            face_min_dis = np.argmin(
                np.sum(((face_dets[:, 2:4] + face_dets[:, :2]) / 2. - center_of_the_face) ** 2, axis=1))

            face_bbox = face_dets[face_min_dis][:4]
            face_prob = face_dets[face_min_dis][4]


            face_image = rgb_img[int(face_bbox[1]): int(face_bbox[3]), int(face_bbox[0]): int(face_bbox[2])]

            #cv2.imwrite('/home/jiasong/centerface/prj-python/' + '%d.jpg' % i, face_image)

            [h, w, c] = face_image.shape

            box = np.array(
                [0, face_image.shape[1] - 1, 0, face_image.shape[0] - 1])  # cropped with bounding box

            pos = face_3d_model.process(face_image, box)

            vertices = face_3d_model.get_vertices(pos)
            save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

            kpt = face_3d_model.get_landmarks(pos)

            bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            sparse_face = plot_kpt(bgr_face_image, kpt)

            dense_face = plot_vertices(bgr_face_image, vertices)
            image[int(face_bbox[1]): int(face_bbox[3]), int(face_bbox[0]): int(face_bbox[2])] = cv2.resize(
                sparse_face, (w, h))

            
            for kpt_elem in kpt:
                kpt_elem[0] +=face_bbox[0]
                kpt_elem[1] +=face_bbox[1]

            face_keypoints = kpt[:,:2]

            person['FaceKeypoint'] = face_keypoints 

            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            
        i += 1
    return result
