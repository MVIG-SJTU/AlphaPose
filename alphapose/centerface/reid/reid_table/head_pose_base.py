import collections
import copy
import logging
from typing import List

import numpy as np
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment

from .base_idbase import BaseReidDatabase
from .reid_utils import HEADFLAGS, MatchFlags, assign_head_status

logger = logging.getLogger(__name__)


class ReidDataBase(BaseReidDatabase):
    def __init__(self, config):
        BaseReidDatabase.__init__(self)
        self.config = copy.deepcopy(config)
        # max number of features in each cluster
        self.num_feature_in_each_cluster = len(HEADFLAGS)
        self.max_feature_in_each_cluster = self.config.CLUSTER_MAX_SIZE
        # for query
        self.threshold_force_store_score = self.config.FORCE_STORE_THRESHOLD
        # if a query feature is with low face socre, then directly ignore it.
        self.threshold_ignore_feature_score = self.config.FACE_SCORE_THRESHOLD
        # if the query matching score is lower than this threshold, drop it.
        self.min_query_matching_score = self.config.MATCHING_THRESHOLD

    def _store_to_reid_table(self, id_dict, face_feature, face_score, face_status):
        wait_for_store = face_feature
        if face_status in id_dict:
            # delete old feature
            if len(id_dict[face_status]) > self.num_feature_in_each_cluster:
                all_features = np.array([v[0] for v in id_dict[face_status]])
                compare_matrix = np.matmul(all_features, np.transpose(all_features))
                compare_matrix = np.minimum(compare_matrix, 1 - np.eye(all_features.shape[0]))
                q_index, g_index = np.unravel_index(compare_matrix.argmax(), compare_matrix.shape)
                q_face_score = id_dict[face_status][q_index][1]
                g_face_score = id_dict[face_status][g_index][1]
                if q_face_score > g_face_score:
                    del id_dict[face_status][q_index]
                else:
                    del id_dict[face_status][g_index]

            # add feature
            max_score = id_dict[face_status][0][1]
            max_f = id_dict[face_status][0][0]
            for tup in id_dict[face_status][1:]:
                if tup[1] > max_score:
                    max_score = tup[1]
                    max_f = tup[0]
            if np.sum(wait_for_store * max_f) > self.min_query_matching_score:
                id_dict[face_status].append((wait_for_store, face_score))
            # else:
            #     print('drop features')
        else:
            if face_score > self.threshold_force_store_score:
                id_dict[face_status] = [(wait_for_store, face_score)]

        return id_dict

    def remove(self, id_list):
        for ind in id_list:
            if ind in self.dataset:
                del self.dataset[ind]

    def update(self, all_detections):
        """
        :param all_detections:  list of list of Detection
        :return:
        """
        for detection in all_detections:
            # for detection in detection_group:
            target_id = detection.target_id
            face_score = detection.face_score
            face_feature = detection.face_feature
            if len(face_feature.shape) > 1:
                face_feature = np.squeeze(face_feature)
            face_status = assign_head_status(detection.face_angle)
            # ignore all features whose face score is lower than self.threshold_ignore_feature_score
            if face_score < self.threshold_ignore_feature_score:
                continue
            # add good face features to the reid table
            if target_id in self.dataset:
                self.dataset[target_id] = self._store_to_reid_table(self.dataset[target_id], face_feature,
                                                                    face_score, face_status)
            else:
                returned_dict = self._store_to_reid_table({}, face_feature, face_score, face_status)
                if bool(returned_dict):
                    self.dataset[target_id] = returned_dict

    def _get_linear_assignment_results(self, dist_matrix, query_ids, gallery_ids):
        # get matched id
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        match_ids = [None] * len(query_ids)
        for i in range(row_ind.shape[0]):
            match_ids[row_ind[i]] = gallery_ids[col_ind[i]]
        # get matched scores
        match_distances = [1.] * len(query_ids)
        for i in range(row_ind.shape[0]):
            match_distances[row_ind[i]] = dist_matrix[row_ind[i], col_ind[i]]
        match_similarity = [1 - dist for dist in match_distances]
        for i in range(len(match_similarity)):
            if match_similarity[i] >= 2:
                match_ids[i] = None
                match_similarity[i] = 0.

        duplicate_ids = [item for item, count in collections.Counter(match_ids).items() if count > 1]
        for dup_id in duplicate_ids:
            indexes = list(np.where(np.array(match_ids) == dup_id)[0])
            duplicate_patch_scores = np.array(match_similarity)[indexes]
            max_score_index = list(duplicate_patch_scores).index(max(list(duplicate_patch_scores)))
            del indexes[max_score_index]
            for index in indexes:
                match_ids[index] = None
                match_similarity[index] = 0.

        return np.array(match_ids), np.array(match_similarity)

    def _get_gallery_ids(self, ignore_ids: List):
        gallery_ids = set(self.dataset.keys()) - set(ignore_ids)
        return list(gallery_ids)

    # search all persons in one frame
    def _retrieval(self, in_features, in_cameras, raw_gallery_ids):
        """Computes and returns closest entity based on features
        Args:
           features: List[M_i x list[L-dimensional np.float32 vector]]
           cameras: List[M_i x np.float32 vetor]
           raw_gallery_ids: List of gallery ids.
        Returns:
           hash_ids(list): list of ids, id could be none
        """
        gallery_features = []
        gallery_ids = []
        for id_ in raw_gallery_ids:
            id_value = self.dataset[id_]
            for store_head_status, cam_value in id_value.items():
                # if store_head_status == in_head_status:
                average_f = np.sum([f[0] for f in cam_value], axis=0)
                average_f /= (LA.norm(average_f, axis=0, keepdims=True, ord=2) + 1e-12)
                gallery_ids.append(id_)
                gallery_features.append(average_f)

        # gallery is empty. return directly
        if len(gallery_features) == 0:
            return [None] * len(in_features), [MatchFlags.EMPTYDICT] * len(in_features)
        gallery_features = np.array(gallery_features).transpose()

        # build query feature matrix
        query_index_in_input_list = []  # store temporary ids [0~N]
        query_features = []
        for i, value in enumerate(in_features):
            value = np.array(value)
            related_cameras = np.array(in_cameras[i])
            unique_cam = np.unique(related_cameras)
            for cam in unique_cam:
                related_f = value[related_cameras == cam]
                average_f = np.sum(related_f, axis=0)
                average_f /= LA.norm(average_f, axis=0, keepdims=True, ord=2)
                query_index_in_input_list.append(i)
                query_features.append(average_f)
        if len(query_features) == 0:
            return [], []
        query_features = np.array(query_features)

        # get similarity matrix
        similarity_matrix = np.matmul(query_features, gallery_features)  # M * N
        similarity_matrix = np.minimum(similarity_matrix, 1.0)

        # print similarity matrix
        def __print_similarity(matrix, gallery_name):
            gallery_name = [str(x)[-4:] for x in gallery_name]
            form = 'ID: {}' + '  {}' * (len(gallery_name) - 1)
            logging.debug('----- query with ReID -----')
            logging.debug(form.format(*gallery_name))
            matrix = list(matrix)
            for line in matrix:
                form = '   {:.3f}' + ' {:.3f}' * (len(line) - 1)
                logging.debug(form.format(*list(line)))

        __print_similarity(similarity_matrix, gallery_ids)
        match_ids, match_scores = self._get_linear_assignment_results(1 - similarity_matrix, query_index_in_input_list,
                                                                      gallery_ids)
        # delete duplicate IDs
        hash_ids = [None] * len(in_features)
        hash_status = [MatchFlags.MATCHED] * len(in_features)
        query_index_in_input_list = np.array(query_index_in_input_list)
        for i, v in enumerate(np.unique(query_index_in_input_list)):
            related_index = query_index_in_input_list == v
            related_id = match_ids[related_index]
            related_score = match_scores[related_index]
            if related_score[np.argmax(related_score)] > self.min_query_matching_score:
                hash_ids[i] = related_id[np.argmax(related_score)]
            else:
                hash_status[i] = MatchFlags.NOTCONVINCED
        # assert no duplicate ids
        assert_ids = list(filter(lambda a: a is not None and a != MatchFlags.NOTCONVINCED, hash_ids))
        assert len(list(np.unique(np.array(assert_ids)))) == len(assert_ids), 'Duplicate IDs from ReID retrival'
        return hash_ids, list(hash_status)

    def reid_query_certain_id(self, target_detections, target_id):
        """
        :param target_detections: a list detection from a single target
        :param target_id: the id you want to check
        :return: list of score
        """
        if target_id not in self.dataset:
            return [None] * len(target_detections)

        stored_info = self.dataset[target_id]
        stored_features = []
        for _, features in stored_info.items():
            average_f = np.sum([f[0] for f in features], axis=0)
            average_f /= (LA.norm(average_f, axis=0, keepdims=True, ord=2) + 1e-12)
            stored_features.append(average_f)
        gallery_features = np.expand_dims(average_f, axis=1)

        query_score = []
        for detection in target_detections:
            if detection.face_score < self.threshold_ignore_feature_score:
                query_score.append(None)
                continue
            # calculate similarities
            face_feature = detection.face_feature
            face_feature /= (LA.norm(face_feature, axis=1, keepdims=True, ord=2) + 1e-12)
            max_score = np.max(np.squeeze(np.matmul(face_feature, gallery_features), axis=0))
            query_score.append(max_score)
        return query_score

    def reid_query_detections(self, reappear_detections, ignored_ids):
        gallery_ids = self._get_gallery_ids(ignored_ids)
        if len(gallery_ids) == 0:
            hash_ids = [None] * len(reappear_detections)
            hash_status = [MatchFlags.EMPTYDICT] * len(reappear_detections)
            return hash_ids, hash_status

        valid_query_indices = []
        features = []
        cameras = []
        face_scores = []

        for i, detection in enumerate(reappear_detections):
            if detection.face_score >= self.threshold_ignore_feature_score:
                f = detection.face_feature
                if len(f.shape) > 1:
                    f = np.squeeze(f)
                features.append(f)
                face_scores.append(detection.face_score)
                cameras.append(detection.camera_id)
                valid_query_indices.append(i)

        result_ids, result_status = self._retrieval(features, cameras, gallery_ids)
        # combine results of bad query features and matching results
        hash_ids = [None for _ in reappear_detections]
        hash_status = [MatchFlags.BADFEATURE for _ in reappear_detections]
        for i, valid_query_index in enumerate(valid_query_indices):
            hash_ids[valid_query_index] = result_ids[i]
            hash_status[valid_query_index] = result_status[i]

        for i, (_, status) in enumerate(zip(hash_ids, hash_status)):
            if status != MatchFlags.MATCHED:
                hash_ids[i] = None
        return hash_ids, hash_status
