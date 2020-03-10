import os
import time

import cv2
import numpy as np


from alphapose.face.centerface import CenterFace
from alphapose.face.prnet import PRN

current_path = os.path.dirname(__file__)

#useless path, enter anything
face_model_path = '../face/models/onnx/centerface.onnx'
face_engine = CenterFace(model_path=face_model_path, landmarks=True)

#useless path, enter anything
face_3d_model_path = '../face/models/prnet.pth'

colors = [tuple(np.random.choice(np.arange(256).astype(np.int32), size=3)) for i in range(100)]


def face_process(opt, result, rgb_img, orig_img, boxes, scores, ids, preds_img, preds_scores):
    device = opt.gpus[0]
    face_3d_model = PRN(face_3d_model_path, device, '../face')
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
           
            for kpt_elem in kpt:
                kpt_elem[0] +=face_bbox[0]
                kpt_elem[1] +=face_bbox[1]

            face_keypoints = kpt[:,:2]

            person['FaceKeypoint'] = face_keypoints 
            
        i += 1
    return result
