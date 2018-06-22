try:
    from torchcv.models.ssd.net import SSD300, SSD512
    from torchcv.models.ssd.box_coder import SSDBoxCoder

    from torchcv.models.fpnssd.net import FPNSSD512
    from torchcv.models.fpnssd.box_coder import FPNSSDBoxCoder

    from torchcv.models.retinanet.net import RetinaNet
    from torchcv.models.retinanet.box_coder import RetinaBoxCoder
except ImportError:
    from ssd.torchcv.models.ssd.net import SSD300, SSD512
    from ssd.torchcv.models.ssd.box_coder import SSDBoxCoder

    from ssd.torchcv.models.fpnssd.net import FPNSSD512
    from ssd.torchcv.models.fpnssd.box_coder import FPNSSDBoxCoder

    from ssd.torchcv.models.retinanet.net import RetinaNet
    from ssd.torchcv.models.retinanet.box_coder import RetinaBoxCoder
