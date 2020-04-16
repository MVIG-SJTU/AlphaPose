from .efficientdet import EfficientDet


def get_efficientdet(num_layers, cfg):
    model = EfficientDet(intermediate_channels=cfg.MODEL.INTERMEDIATE_CHANNEL)
    return model
