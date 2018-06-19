try:
    from torchcv.models.fpnssd.net import FPNSSD512

    from torchcv.models.retinanet.box_coder import BoxCoder
    from torchcv.models.retinanet.retinanet import RetinaNet

    from torchcv.models.ssd.net import SSD300, SSD512
    from torchcv.models.ssd.box_coder import SSDBoxCoder
except ImportError:
    from ssd.torchcv.models.fpnssd.net import FPNSSD512

    from ssd.torchcv.models.retinanet.box_coder import BoxCoder
    from ssd.torchcv.models.retinanet.retinanet import RetinaNet

    from ssd.torchcv.models.ssd.net import SSD300, SSD512
    from ssd.torchcv.models.ssd.box_coder import SSDBoxCoder
