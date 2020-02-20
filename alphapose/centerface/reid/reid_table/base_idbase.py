class BaseReidDatabase(object):
    """
    Args:
       dataset: a collection of (hash_id, features) pairs in some format
                (maybe proto?)
    """

    def __init__(self):
        self.dataset = {}

    def update(self, features, cameras, hash_ids):
        """Update dataset with features for a specific hash_id
        Args:
            features: List[M_i x L-dimensional np.float32 array]
            cameras: List[M_i np.float32 vetor]
            hash_ids: List[ids] of length M_i
        """
        raise NotImplementedError

    def get_all_ids(self):
        return ['{:04d}'.format(abs(k) % 10000) for k, v in self.dataset.items()]

    def get_current_table_size(self):
        return len(self.dataset)

    def check_if_in_table(self, new_id):
        return new_id in self.dataset

    # search all persons in one frame
    def retrieval(self, features, cameras, tracked_ids):
        """Computes and returns closest entity based on features
        Args:
           features: List[M_i x L-dimensional np.float32 array]
           cameras: List[M_i np.float32 vetor]
           tracked_ids: List of ids of unknown length, confirmed ids by tracker. ReID should ignore these ids.
        Returns:
           hash_ids(list): list of ids, id could be none
        """
        raise NotImplementedError

    def remove(self, hash_id):
        """Deletes entity with hash_id and all of it's features from the dataset
        Args:
            hash_id(string): unique string identifying the specific person
        """
        raise NotImplementedError
