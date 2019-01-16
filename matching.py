# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

from copy import deepcopy

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])


def candidate_reselect(bboxes, bboxes_scores, pose_preds):

    '''
    Grouping
    '''
    # Group same keypointns together
    kp_groups = grouping(bboxes, bboxes_scores, pose_preds)

    '''
    Re-select
    '''

    # Generate Matrix
    human_num = len(pose_preds.keys())
    costMatrix = []
    for k in range(17):
        kp_group = kp_groups[k]
        joint_num = len(kp_group.keys())

        costMatrix.append(np.zeros((human_num, joint_num)))

    group_size = {k: {} for k in range(17)}

    for n, person in pose_preds.items():
        h_id = n
        assert 0 <= h_id < human_num

        for k in range(17):
            g_id = person['group_id'][k]
            if g_id is not None:
                if g_id not in group_size[k].keys():
                    group_size[k][g_id] = 0

                group_size[k][g_id] += 1

                g_id = int(g_id) - 1
                _, _, score = person[k][0]
                h_score = person['human_score']

                if score < 0.05:
                    costMatrix[k][h_id][g_id] = 0
                else:
                    costMatrix[k][h_id][g_id] = -(h_score * score)

    pose_preds = matching(pose_preds, costMatrix, kp_groups)

    # To JSON
    final_result = []

    for n, person in pose_preds.items():

        final_pose = torch.zeros(17, 2)
        final_score = torch.zeros(17, 1)

        max_score = 0
        mean_score = 0

        xmax, xmin = 0, 1e5
        ymax, ymin = 0, 1e5

        for k in range(17):
            assert len(person[k]) > 0
            x, y, s = person[k][0]

            xmax = max(xmax, x)
            xmin = min(xmin, x)
            ymax = max(ymax, y)
            ymin = min(ymin, y)

            final_pose[k][0] = x.item() - 0.3
            final_pose[k][1] = y.item() - 0.3
            final_score[k] = s.item()
            mean_score += (s.item() / 17)
            max_score = max(max_score, s.item())

        if torch.max(final_score).item() < 0.1:
            continue

        if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < 40 * 40):
            continue

        final_result.append({
            'keypoints': final_pose,
            'kp_score': final_score,
            'proposal_score': mean_score + max_score + person['bbox_score']
        })

    return final_result


def grouping(bboxes, bboxes_scores, pose_preds):

    kp_groups = {}
    for k in range(17):
        kp_groups[k] = {}

    ids = np.zeros(17)

    for n, person in pose_preds.items():
        pose_preds[n]['bbox'] = bboxes[n]
        pose_preds[n]['bbox_score'] = bboxes_scores[n]
        pose_preds[n]['group_id'] = {}
        s = 0

        for k in range(17):
            pose_preds[n]['group_id'][k] = None
            pose_preds[n][k] = np.array(pose_preds[n][k])
            assert len(pose_preds[n][k]) > 0
            s += pose_preds[n][k][0][-1]

        s = s / 17

        pose_preds[n]['human_score'] = s

        for k in range(17):
            latest_id = ids[k]
            kp_group = kp_groups[k]

            assert len(person[k]) > 0
            x0, y0, s0 = person[k][0]
            if s0 < 0.05:
                continue

            for g_id, g in kp_group.items():
                x_c, y_c = kp_group[g_id]['group_center']

                '''
                Get Average Box Size
                '''
                group_area = kp_group[g_id]['group_area']
                group_area = group_area[0] * group_area[1] / (group_area[2] ** 2)

                '''
                Groupingn Criterion
                '''
                # Joint Group
                dist = np.sqrt(
                    ((x_c - x0) ** 2 + (y_c - y0) ** 2) / group_area)

                if dist <= 0.1 * sigmas[k]:  # Small Distance
                    if s0 >= 0.3:
                        kp_group[g_id]['kp_list'][0] += x0 * s0
                        kp_group[g_id]['kp_list'][1] += y0 * s0
                        kp_group[g_id]['kp_list'][2] += s0

                        kp_group[g_id]['group_area'][0] += (person['bbox'][2] - person['bbox'][0]) * person['human_score']
                        kp_group[g_id]['group_area'][1] += (person['bbox'][3] - person['bbox'][1]) * person['human_score']
                        kp_group[g_id]['group_area'][2] += person['human_score']

                        x_c = kp_group[g_id]['kp_list'][0] / kp_group[g_id]['kp_list'][2]
                        y_c = kp_group[g_id]['kp_list'][1] / kp_group[g_id]['kp_list'][2]
                        kp_group[g_id]['group_center'] = (x_c, y_c)

                    pose_preds[n]['group_id'][k] = g_id
                    break
            else:
                # A new keypoint group
                latest_id += 1
                kp_group[latest_id] = {
                    'kp_list': None,
                    'group_center': person[k][0].copy()[:2],
                    'group_area': None
                }

                x, y, s = person[k][0]
                kp_group[latest_id]['kp_list'] = np.array((x * s, y * s, s))

                # Ref Area
                ref_width = person['bbox'][2] - person['bbox'][0]
                ref_height = person['bbox'][3] - person['bbox'][1]
                ref_score = person['human_score']
                kp_group[latest_id]['group_area'] = np.array((
                    ref_width * ref_score, ref_height * ref_score, ref_score))

                pose_preds[n]['group_id'][k] = latest_id
                ids[k] = latest_id
    return kp_groups


def matching(pose_preds, matrix, kp_groups):
    index = []
    for k in range(17):
        human_ind, joint_ind = linear_sum_assignment(matrix[k])
        # human_ind, joint_ind = greedy_matching(matrix[k])

        index.append(list(zip(human_ind, joint_ind)))

    for n, person in pose_preds.items():
        for k in range(17):
            g_id = person['group_id'][k]
            if g_id is not None:
                g_id = int(g_id) - 1
                h_id = n

                x, y, s = pose_preds[n][k][0]
                if ((h_id, g_id) not in index[k]) and len(pose_preds[n][k]) > 1:
                    pose_preds[n][k] = np.delete(pose_preds[n][k], 0, 0)
                elif ((h_id, g_id) not in index[k]) and len(person[k]) == 1:
                    x, y, _ = pose_preds[n][k][0]
                    pose_preds[n][k][0] = (x, y, 1e-5)
                    pass
                elif ((h_id, g_id) in index[k]):
                    x, y = kp_groups[k][g_id + 1]['group_center']
                    s = pose_preds[n][k][0][2]
                    pose_preds[n][k][0] = (x, y, s)

    return pose_preds


def greedy_matching(matrix):
    num_human, num_joint = matrix.shape

    if num_joint <= num_human or True:
        human_ind = np.argmin(matrix, axis=0)
        joint_ind = np.arange(num_joint)
    else:
        pass

    return human_ind.tolist(), joint_ind.tolist()
