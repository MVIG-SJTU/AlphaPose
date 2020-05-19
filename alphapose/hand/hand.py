import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

from alphapose.hand.hand_src import Hand
from alphapose.hand import model
from alphapose.hand import util

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

current_path = os.path.dirname(__file__)
model_path = current_path + '/models/hand_pose_model.pth'
hand_estimation = Hand(model_path)

def draw_handpose(canvas, all_hand_peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    fig = Figure(figsize=plt.figaspect(canvas))

    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)
    old_hand_peaks = all_hand_peaks.copy()
    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
    all_hand_peaks = []
    all_hand_peaks.append(old_hand_peaks)
    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1)==0)==0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            ax.plot(x, y, 'r.')
            if show_number:
                ax.text(x, y, str(i))
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas


# candidate is all coordinates, subset are n*17 with the index in candidate
def handDetect(result, orig_img):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = orig_img.shape[0:2]
    # for person in subset.astype(int):
    for person in result:
        keypoints = person['keypoints']
        kp_score = person['kp_score']
        # print(keypoints)
        format_keypoints = keypoints.numpy()
        format_kp_score = kp_score.numpy()
        format_kp_score = format_kp_score.flatten()
        # print(format_keypoints)
        # # if any of three not detected
        has_left = np.sum(format_kp_score[[5, 7, 9]] < 0.35) == 0
        # print("has_left:", has_left)
        has_right = np.sum(format_kp_score[[6, 8, 10]] < 0.35) == 0
        # print("has_right:", has_right)
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            # left_shoulder_index, left_elbow_index, left_wrist_index = format_keypoints[[5, 7, 9]]
            x1, y1 = format_keypoints[5][:2]
            x2, y2 = format_keypoints[7][:2]
            x3, y3 = format_keypoints[9][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            # right_shoulder_index, right_elbow_index, right_wrist_index = format_keypoints[[6, 8, 10]]
            x1, y1 = format_keypoints[6][:2]
            x2, y2 = format_keypoints[8][:2]
            x3, y3 = format_keypoints[10][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:

            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

        '''
        return value: [[x, y, w, True if left hand else False]].
        width=height since the network require squared input.
        x, y is the coordinate of top left 
        '''

        canvas = orig_img
        all_hand_peaks = []
        for x, y, w, is_left in detect_result:

            peaks = hand_estimation(orig_img[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            
            # all_hand_peaks.append(peaks)
            # all_hand_peaks is for all hands in one image, but now is done person by person, not the whole image.
        all_hand_peaks = peaks

        person['HandKeypoint'] = all_hand_peaks 
        # print('write',all_hand_peaks)
        canvas = draw_handpose(canvas, all_hand_peaks)
        # canvas = draw_handpose(canvas, all_hand_peaks, 1)
        cv2.imwrite('/home/jiasong/pytorch-openpose/res/res.jpg', canvas)

    return result


