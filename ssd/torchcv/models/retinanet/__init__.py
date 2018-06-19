try:
    from torchcv.models.retinanet.fpn import FPN50
    from torchcv.models.retinanet.box_coder import BoxCoder
    from torchcv.models.retinanet.retinanet import RetinaNet
except ImportError:
    from ssd.torchcv.models.retinanet.fpn import FPN50
    from ssd.torchcv.models.retinanet.box_coder import BoxCoder
    from ssd.torchcv.models.retinanet.retinanet import RetinaNet
