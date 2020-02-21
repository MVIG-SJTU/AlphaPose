from .reid_table.head_pose_base import ReidDataBase


class ReIDManager(object):
    def __init__(self, config):
        self.reid_table = ReidDataBase(config)

    def query_targets(self, reappear_targets, ignored_targets):
        if len(reappear_targets) == 0:
            return [], []
        reappear_detections = []
        for single_target in reappear_targets:
            best_detection = self._get_detection_with_highest_face(single_target)
            reappear_detections.append(best_detection)

        ignored_id = [t.id for t in ignored_targets]
        hash_ids, hash_status = self.reid_table.reid_query_detections(reappear_detections, ignored_id)
        return hash_ids, hash_status

    def update_targets(self, tracked_targets):
        all_detections = [self._get_latest_detection(target) for target in tracked_targets]
        # update reid features
        self.reid_table.update(all_detections)

    def remove_targets(self, removed_targets):
        will_remove_ids = [target.id for target in removed_targets]
        self.reid_table.remove(will_remove_ids)

    def query_certain_id(self, detection_list, target_id):
        if len(detection_list) == 0:
            return []
        return self.reid_table.reid_query_certain_id(detection_list, target_id)

    def _get_detection_with_highest_face(self, target):
        detection_list = target.last_detections  # list(target.get_detections())
        detection_list = sorted(detection_list, key=lambda t: t.face_score)[::-1]
        return detection_list[0]

    def _get_latest_detection(self, target):
        # target.last_detections: a list of last detections sorted by time (currently contains 4 detections)
        return target.last_detections[0]
