from .coco_det import Mscoco_det
from .concat_dataset import ConcatDataset
from .custom import CustomDataset
from .mscoco import Mscoco
from .mscoco_with_foot import Mscoco_with_foot
from .coco_det_with_foot import Mscoco_det_with_foot
from .mpii import Mpii

__all__ = ['CustomDataset', 'Mscoco', 'Mscoco_det', 'Mpii', 'ConcatDataset', 'Mscoco_with_foot', 'Mscoco_det_with_foot']
