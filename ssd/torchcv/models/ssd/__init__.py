try:
    from torchcv.models.ssd.net import SSD300, SSD512
    from torchcv.models.ssd.box_coder import SSDBoxCoder
except ImportError:
    from ssd.torchcv.models.ssd.net import SSD300, SSD512
    from ssd.torchcv.models.ssd.box_coder import SSDBoxCoder
