import numpy as np
from mxnet import nd

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 1.3
gamma = 22
scoreThreds = 0.05
matchThreds = 4
alpha = 0.1


def pose_nms(pose_preds, pose_scores, bbox_preds, bbox_scores, ori_bbox):
    '''
    Parametric Pose NMS algorithm
    pose_preds:     pose locations nd array (n, 17, 2)
    pose_scores:    pose scores nd array    (n, 17, 1)

    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n,)

    return:
        pred_coords:    pose locations array    (n_new, 17, 2)
        confidence:     pose scores array       (n_new, 17, 1)
    '''
    np_bbox_preds = np.array(bbox_preds)
    np_ori_bbox = np.array(ori_bbox)
    np_bbox_scores = np.array(bbox_scores)
    pose_preds = pose_preds.asnumpy()
    pose_scores = pose_scores.asnumpy()

    pose_scores[pose_scores < scoreThreds] == 1e-5

    ori_pose_preds = pose_preds.copy()
    ori_pose_scores = pose_scores.copy()
    ori_bbox = np_ori_bbox.copy()

    final_result = []
    pred_coords, confidence, pred_bbox = [], [], []

    xmax = np_bbox_preds[:, 2]
    xmin = np_bbox_preds[:, 0]
    ymax = np_bbox_preds[:, 3]
    ymin = np_bbox_preds[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = np_bbox_preds.shape[0]

    human_scores = np.mean(pose_scores, axis=1) + np.max(pose_scores, axis=1) + np_bbox_scores

    human_ids = np.arange(nsamples)

    # Do pPose-NMS
    pick = []
    merge_ids = []
    while(human_scores.shape[0] != 0):
        pick_id = np.argmax(human_scores)
        pick.append(human_ids[pick_id])

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[pick_id]]
        simi = get_parametric_distance(
            pick_id, pose_preds, pose_scores, ref_dist)
        num_match_keypoints = PCK_match(
            pose_preds[pick_id], pose_preds, ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        delete_ids = np.arange(human_scores.shape[0])[
            (simi > gamma) | (num_match_keypoints >= matchThreds)]

        if delete_ids.shape[0] == 0:
            delete_ids = pick_id

        merge_ids.append(human_ids[delete_ids])
        pose_preds = np.delete(pose_preds, delete_ids, axis=0)
        pose_scores = np.delete(pose_scores, delete_ids, axis=0)
        human_ids = np.delete(human_ids, delete_ids)
        human_scores = np.delete(human_scores, delete_ids, axis=0)

    assert len(merge_ids) == len(pick)
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    ori_bbox_pick = ori_bbox[pick]

    for j in range(len(pick)):
        ids = np.arange(17)
        max_score = np.max(scores_pick[j, ids, 0])

        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j], ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick[j]])

        max_score = np.max(merge_score[ids])

        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])

        if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < 40 * 40):
            continue

        pred_coords.append(nd.array(merge_pose[None, :, :]))
        confidence.append(nd.array(merge_score[None, :, :]))
        pred_bbox.append(nd.array(ori_bbox_pick[j][None, :]))

        final_result.append({
            'keypoints': merge_pose,
            'kp_score': merge_score,
            'proposal_score': np.mean(merge_score) + 1.25 * np.max(merge_score)
        })
    if len(pred_coords) == 0:
        return None, None, None
    pred_coords = nd.concatenate(pred_coords)
    confidence = nd.concatenate(confidence)
    pred_bbox = nd.concatenate(pred_bbox)
    return pred_coords, confidence, pred_bbox


def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]

    dist = np.linalg.norm(pick_preds[None, :] - all_preds, axis=2)
    mask = (dist <= 1)

    # Define a keypoints distance
    score_dists = np.zeros((all_preds.shape[0], 17))
    keypoint_scores = keypoint_scores.squeeze()

    if keypoint_scores.ndim == 1:
        keypoint_scores = keypoint_scores[None, :]
    if pred_scores.ndim == 1:
        pred_scores = pred_scores[:, None]

    # The predicted scores are repeated up to do broadcast
    pred_scores = np.tile(pred_scores, (1, all_preds.shape[0])).transpose(1, 0)
    score_dists[mask] = np.tanh(pred_scores[mask] / delta1) * np.tanh(keypoint_scores[mask] / delta1)

    point_dist = np.exp((-1) * dist / delta2)
    final_dist = np.sum(score_dists, axis=1) + mu * np.sum(point_dist, axis=1)

    return final_dist


def PCK_match(pick_pred, all_preds, ref_dist):
    dist = np.linalg.norm(pick_pred[None, :] - all_preds, axis=2)
    ref_dist = min(ref_dist, 7)

    num_match_keypoints = np.sum(dist / ref_dist <= 1, axis=1)

    return num_match_keypoints


def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    '''

    dist = np.linalg.norm(ref_pose[None, :] - cluster_preds, axis=2)

    kp_num = 17
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = np.zeros((kp_num, 2))
    final_score = np.zeros(kp_num)

    if cluster_preds.ndim == 2:
        cluster_preds = cluster_preds[None, :, :]
        cluster_scores = cluster_scores[None, :, :]
    if mask.ndim == 1:
        mask = mask[None, :]

    # Weighted Merge
    masked_scores = cluster_scores * mask.astype(np.float32)[:, :, None]
    normed_scores = masked_scores / np.sum(masked_scores, axis=0)

    final_pose = (cluster_preds * np.tile(normed_scores, (1, 1, 2))).sum(axis=0)
    final_score = (masked_scores * normed_scores).sum(axis=0)

    return final_pose, final_score
