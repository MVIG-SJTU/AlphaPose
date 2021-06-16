""" PyTorch EfficientDet support benches

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
from .anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS


def _post_process(config, cls_outputs, box_outputs):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        config: a parameter dictionary that includes `min_level`, `max_level`,  `batch_size`, and `num_classes`.

        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([
        cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, config.num_classes])
        for level in range(config.num_levels)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
        for level in range(config.num_levels)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=MAX_DETECTION_POINTS)
    indices_all = cls_topk_indices_all // config.num_classes
    classes_all = cls_topk_indices_all % config.num_classes

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, config.num_classes))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


class DetBenchEval(nn.Module):
    def __init__(self, model, config, nms_thres=0.5, max_dets=100):
        super(DetBenchEval, self).__init__()
        self.config = config
        self.nms_thres = nms_thres
        self.max_dets = max_dets
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)

    def forward(self, x, image_scales):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)

        batch_detections = []
        # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], self.anchors.boxes, indices[i], classes[i], image_scales[i], 
                nms_thres=self.nms_thres, max_dets=self.max_dets
                )
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)


class DetBenchTrain(nn.Module):
    def __init__(self, model, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
        self.model = model
        anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(anchors, config.num_classes, match_threshold=0.5)
        self.loss_fn = None

    def forward(self, x, gt_boxes, gt_labels):
        class_out, box_out = self.model(x)
        loss = None
        gcl = []
        gbl = []
        total_positive = 0
        # FIXME the per-sample organization of reference code less than desirable, should change to batched
        for i in range(x.shape[0]):
            gt_class_out, gt_box_out, num_positive = self.anchor_labeler.label_anchors(gt_boxes[i], gt_labels[i])
            gcl.append(gt_class_out)
            gbl.append(gt_box_out)
            total_positive += num_positive

        # FIXME compute loss

        return loss
