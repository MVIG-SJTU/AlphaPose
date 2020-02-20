import os
import time
from threading import Thread
from queue import Queue
import math

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms

from alphapose.centerface.centerface import CenterFace
from alphapose.centerface.prnet import PRN
from alphapose.centerface.utils.cv_plot import plot_kpt, plot_pose_box, plot_vertices
from alphapose.centerface.utils.render_app import get_visibility, get_uv_mask, get_depth_image
from alphapose.centerface.utils.estimate_pose import estimate_pose


DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

current_path = os.path.dirname(__file__)

face_model_path = '../centerface/models/onnx/centerface.onnx'
face_engine = CenterFace(model_path=face_model_path, landmarks=True)

#useless path, enter anything
face_3d_model_path = '../centerface/models/prnet.pth'

face_3d_model = PRN(face_3d_model_path, '../centerface')

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


def add_coco_bbox(image, bbox, cat, conf=1, color=None):
    cat = int(cat)
    txt = '{}{:.1f}'.format('person', conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (int(color[0]), int(color[1]), int(color[2])), 2)
    cv2.putText(image, txt, (bbox[0], bbox[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def add_coco_hp(image, points, color):
    print(points)
    for j in range(17):
        print(points[j, 0])
        print(points[j, 1])
        print(type(image))
        cv2.circle(image, (points[j, 0], points[j, 1]), 2, (int(color[0]), int(color[1]), int(color[2])), -1)

    stickwidth = 2
    cur_canvas = image.copy()
    for j, e in enumerate(_kp_connections):
        if points[e].min() > 0:
            X = [points[e[0], 1], points[e[1], 1]]
            Y = [points[e[0], 0], points[e[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, (int(color[0]), int(color[1]), int(color[2])))
            image = cv2.addWeighted(image, 0.5, cur_canvas, 0.5, 0)

    return image


class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.final_result = []
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
            self.final_result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)
            self.final_result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_track:
            from PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitely
        while True:
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                self.wait_and_put(self.final_result_queue, None)
                if self.save_video:
                    stream.release()
                return

            rgb_img = orig_img

            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                pred = hm_data.cpu().data.numpy()
                assert pred.ndim == 4

                if hm_data.size()[1] == 49:
                    self.eval_joints = [*range(0, 49)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(pred[i][self.eval_joints], bbox)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                result = pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

                ### jiasong update 2.20
                if self.opt.show_face:
                    boxes = boxes.numpy()
                    
                    i = 0
                    face_engine.transform(orig_img.shape[0], orig_img.shape[1])
                    face_dets, lms = face_engine(orig_img, threshold=0.35)
                    #face_dic={}
                    #face_dic['facebox']=face_dets

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
                            person['facepoint'] = kpt
                            camera_matrix, pose = estimate_pose(vertices)

                            bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                            image_pose = plot_pose_box(bgr_face_image, camera_matrix, kpt)
                            sparse_face = plot_kpt(bgr_face_image, kpt)

                            dense_face = plot_vertices(bgr_face_image, vertices)
                            image[int(face_bbox[1]): int(face_bbox[3]), int(face_bbox[0]): int(face_bbox[2])] = cv2.resize(
                                sparse_face, (w, h))


                        #image_save_path = os.path.join(save_image_dir, str(image_count) + '.png')
                        #cv2.imwrite('/home/jiasong/centerface/prj-python/good.jpg', image)




                        i += 1

                    result = {
                        'imgname': im_name,
                        'result': result,
                        'facebox': face_dets
                    }
               #     print(result)
                else:
                    result = {
                        'imgname': im_name,
                        'result': result
                    }

                if self.opt.pose_track:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']
                self.wait_and_put(self.final_result_queue, result)
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, add_bbox=(self.opt.pose_track | self.opt.tracking))
                    self.write_image(img, im_name, stream=stream if self.save_video else None)

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.commit()
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        self.commit()
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        while True:
            final_res = self.wait_and_get(self.final_result_queue)
            if final_res:
                self.final_result.append(final_res)
            else:
                break
        self.result_worker.join()

    def clear_queues(self):
        self.clear(self.result_queue)
        self.clear(self.final_result_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def commit(self):
        # commit finished final results to main process
        while not self.final_result_queue.empty():
            self.final_result.append(self.wait_and_get(self.final_result_queue))

    def results(self):
        # return final result
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

