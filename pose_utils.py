import numpy as np

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThresh = 0.3
matchThresh = 5
areaThresh = 0  # 40 * 40.5
alpha = 0.1


def pose_nms(boxes, box_scores, pose_coords, pose_scores):
    '''
    Parametric pose NMS
    INPUT:
        boxes: pose bounding boxes, (N, 4)
        box_scores: bounding box scores, (N,)
        pose_coords: pose coordinates, (N, 17, 2)
        pose_scores: pose scores, (N, 17, 1)
    OUTPUT:
        final_result: list of json results, {
            'keypoints': (17, 2)
            'kp_score': (17, )
            'proposal_score': float32
        }
    '''
    xmin, xmax = boxes[:, 0], boxes[:, 2]
    ymin, ymax = boxes[:, 1], boxes[:, 3]
    width = xmax - xmin
    height = ymax - ymin
    ref_dists = alpha * np.maximum(width, height)

    num_boxes = boxes.shape[0]
    human_scores = pose_scores.mean(axis=1)[:, 0]
    human_idxs = np.arange(num_boxes)

    # keypoint distances
    kp_dists = keypoint_distance(pose_coords)
    # score similarities
    score_simi = score_similarity(pose_scores)
    # pair numbers of matched points
    num_match_keypoints = PCK_match(kp_dists, ref_dists)
    # parametric similarities
    param_simi = parametric_similarity(kp_dists, score_simi, ref_dists)

    picked_idxs = []
    merged_idxs = []

    while human_scores.shape[0] != 0:
        # select pose with the highest score
        picked_idx = np.argmax(human_scores)
        picked_idxs.append(human_idxs[picked_idx])

        # select poses to be deleted
        mask1 = (param_simi[human_idxs[picked_idx], human_idxs] > gamma)
        mask2 = (num_match_keypoints[human_idxs[picked_idx], human_idxs] >= matchThresh)
        deleted_idxs = np.where(mask1 | mask2)[0]
        if deleted_idxs.shape[0] == 0:
            deleted_idxs = picked_idx

        # update pose list
        merged_idxs.append(human_idxs[deleted_idxs])
        human_idxs = np.delete(human_idxs, deleted_idxs)
        human_scores = np.delete(human_scores, deleted_idxs)

    assert len(merged_idxs) == len(picked_idxs)
    picked_scores = pose_scores[picked_idxs]
    picked_box_scores = box_scores[picked_idxs]
    picked_box = boxes[picked_idxs]

    # pose merge
    final_result = []
    for i in range(len(picked_idxs)):
        if np.all(picked_scores[i] < scoreThresh):
            continue
        picked_idx = picked_idxs[i]
        merged_idx = merged_idxs[i]
        merged_coords, merged_scores = pose_merge(kp_dists[picked_idx, merged_idx],
                                                  pose_coords[merged_idx],
                                                  pose_scores[merged_idx],
                                                  ref_dists[picked_idx])

        if np.all(merged_scores < scoreThresh):
            continue

        if (1.5 ** 2 * np.prod(np.ptp(merged_coords, axis=0)) < areaThresh):
            continue

        final_result.append({
            'keypoints': merged_coords - 0.3,
            'kp_score': merged_scores,
            'proposal_score': np.mean(merged_scores) + picked_box_scores[i] + 1.25 * max(merged_scores)
        })
    return final_result, picked_box, picked_box_scores


def pose_merge(dists, cluster_coords, cluster_scores, ref_dist):
    '''
    Merge poses in the same cluster
    INPUT:
        dists: slice of distance matrix, (n, 17)
        cluster_coords: pose coords in one cluster, (n, 17, 2)
        cluster_scores: pose scores in one cluster, (n, 17, 1)
        ref_dist: reference distance of target pose, float32
    OUTPUT:
        merged_coords: merged pose coordinates, (17, 2)
        merged_scores: merged pose scores, (17,)
    '''
    ref_dist = min(ref_dist, 15)
    mask = np.expand_dims((dists < ref_dist), axis=-1)

    # get pose weight
    masked_scores = cluster_scores * mask
    normed_scores = masked_scores / np.sum(masked_scores, axis=0, keepdims=True)

    # weighted merge
    merged_coords = (cluster_coords * normed_scores).sum(axis=0)
    merged_scores = (masked_scores * normed_scores).sum(axis=0)

    return merged_coords, merged_scores


def parametric_similarity(kp_dists, score_simi, ref_dists):
    '''
    Compute parametric similarity
    INPUT:
        kp_dists: keypoint distance matrix, (N, N, 17)
        score_simi: score similarity matrix, (N, N, 17)
        ref_dists: reference distance of target poses, (N,)
    OUTPUT:
        param_simi: parametric similarity matrix, (N, N)
    '''
    mask = (kp_dists <= 1).astype(np.float32)
    kp_simi = np.exp((-1) * kp_dists / delta2)
    param_simi = mask * score_simi + mu * kp_simi
    param_simi = np.sum(param_simi, axis=2)
    return param_simi


def PCK_match(kp_dists, ref_dists):
    '''
    find matched keypoints based on refrence distance
    INPUT:
        kp_dists: keypoint distance matrix, (N, N, 17)
        ref_dists: reference distance of target poses (N,)
    OUTPUT:
        num_match_keypoints: number of matched keypoints, (N, N)
    '''
    ref_dists = ref_dists.reshape([-1, 1, 1])
    num_match_keypoints = np.sum(kp_dists < ref_dists, axis=2)
    return num_match_keypoints


def keypoint_distance(pose_coords):
    '''
    Compute keypoint distance matrix
    INPUT:
        pose_coords: pose coordinates, (N, 17, 2)
    OUTPUT:
        dists: keypoint distance matrix, (N, N, 17)
    '''
    A = pose_coords.transpose([1, 0, 2])
    B = pose_coords.transpose([1, 2, 0])
    A2 = np.sum(np.square(A), axis=2, keepdims=True)
    B2 = np.sum(np.square(B), axis=1, keepdims=True)
    AB = np.matmul(A, B)
    dists = A2 + B2 - 2 * AB
    dists = dists.transpose([1, 2, 0])
    return dists


def score_similarity(pose_scores):
    '''
    Compute score similarity matrix
    INPUT:
        pose_scores: pose scores, (N, 17, 1)
    OUTPUT:
        simi: score simiarity matrix, (N, N, 17)
    '''
    A = pose_scores.transpose([1, 0, 2])
    B = pose_scores.transpose([1, 2, 0])
    simi = np.tanh(A / delta1) * np.tanh(B / delta1)
    simi = simi.transpose([1, 2, 0])
    return simi
