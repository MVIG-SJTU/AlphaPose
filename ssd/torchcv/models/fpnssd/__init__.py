try:
    from torchcv.models.fpnssd.net import FPNSSD512
except ImportError:
    from ssd.torchcv.models.fpnssd.net import FPNSSD512
