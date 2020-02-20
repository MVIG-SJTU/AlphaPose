from enum import Enum


class MatchFlags(Enum):
    MATCHED = 0
    BADFEATURE = 1
    NOTCONVINCED = 2
    FIRSTTIME = 3
    EMPTYDICT = 4
    UNREGISTER = 5
    NODETECTION = 6


class HEADFLAGS(Enum):
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3
    S4 = 4
    S5 = 5
    S6 = 6


# Data template
class Detection:
    def __init__(self, ind, score, camera, feature, landmarks, bbox):
        self.target_id = ind
        self.face_score = score
        self.features = feature
        self.camera_id = camera
        self.landmarks = landmarks
        self.bbox = bbox

    def set_new_id(self, new_id):
        self.target_id = new_id

    def get_id(self):
        return self.target_id

    def get_face_score(self):
        return self.face_score


def assign_head_status(yaw):
    # if abs(yaw) > 25:
    #     head_status = HEADFLAGS.S2
    # elif abs(yaw) > 15:
    #     head_status = HEADFLAGS.S1
    # else:
    #     head_status = HEADFLAGS.S0
    if abs(yaw) > 30:
        head_status = HEADFLAGS.S6
    elif abs(yaw) > 25:
        head_status = HEADFLAGS.S5
    elif abs(yaw) > 20:
        head_status = HEADFLAGS.S4
    elif abs(yaw) > 15:
        head_status = HEADFLAGS.S3
    elif abs(yaw) > 10:
        head_status = HEADFLAGS.S2
    elif abs(yaw) > 5:
        head_status = HEADFLAGS.S1
    else:
        head_status = HEADFLAGS.S0
    return head_status
