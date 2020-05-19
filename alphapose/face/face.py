import os
import time

import cv2
import numpy as np


from alphapose.face.centerface import CenterFace
from alphapose.face.prnet import PRN

#useless path, enter anything
#face_model_path = '../face/models/onnx/centerface.onnx'
#face_engine = CenterFace(model_path=face_model_path, landmarks=True)

#useless path, enter anything
#face_3d_model_path = '../face/models/prnet.pth'
#face_3d_model = PRN(face_3d_model_path, device, '../face')


def face_process(face_3d_model, result, orig_img):
    face_engine = CenterFace( landmarks=True)
    rgb_img = orig_img[:, :, ::-1]
    [H, W, _] = rgb_img.shape
    face_engine.transform(orig_img.shape[0], orig_img.shape[1])
    face_dets, lms = face_engine(orig_img, threshold=0.35)

    result_new = []

    for person in result:

        keypoints = person['keypoints']
        kp_score = person['kp_score']
      
        keypoints = keypoints.numpy()
        kp_score = kp_score.numpy()


        center_of_the_face = np.mean(keypoints[:5, :], axis=0)
        face_conf = np.mean(kp_score[:5, :], axis=0)

        face_keypoints = -1*np.ones((68,3))
        if face_conf > 0.5 and len(face_dets) > 0:
            face_min_dis = np.argmin(
                np.sum(((face_dets[:, 2:4] + face_dets[:, :2]) / 2. - center_of_the_face) ** 2, axis=1))

            face_bbox = face_dets[face_min_dis][:4]
            face_prob = face_dets[face_min_dis][4]
            if center_of_the_face[0] < face_bbox[0] or center_of_the_face[1] < face_bbox[1] or center_of_the_face[0] > face_bbox[2] or center_of_the_face[1] > face_bbox[3]:
                continue 
            if face_prob < 0.5:
                continue

            ## below is by intuitive box
            # wid = max(keypoints[:5, 0]) - min(keypoints[:5, 0])
            # hgt = max(keypoints[:5, 1]) - min(keypoints[:5, 1])
            # face_bbox = [max(0,min(keypoints[:5, 0])-0.1*wid), max(0,min(keypoints[:5, 1])-2*hgt), min(max(keypoints[:5, 0])+0.1*wid,W),  min(H,max(keypoints[:5, 1])+2.5*hgt)]
            # print(face_bbox)
            


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

            face_keypoints = kpt[:,:3]
        # print('face', face_keypoints)
        person['FaceKeypoint'] = face_keypoints 
        result_new.append(person)            
    return result_new
