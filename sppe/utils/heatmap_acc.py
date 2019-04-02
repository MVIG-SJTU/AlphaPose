import numpy as np
import mxnet as mx
from mxnet.metric import check_label_shapes
from mxnet import nd


class HeatmapAccuracy(mx.metric.EvalMetric):
    """ Computes accuracy classification score with optional ignored labels. """

    def __init__(self, axis=1, name='heatmap_accuracy', hm_type='gaussian', threshold=0.5,
                 output_names=None, label_names=None, ignore_labels=None):
        super(HeatmapAccuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis
        self.ignore_labels = np.array(ignore_labels).flatten()
        self.sum_metric = 0
        self.num_inst = 0
        self.hm_type = hm_type
        self.threshold = threshold

    def _calc_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))

        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1

        return dists

    def _dist_acc(self, dists):
        dist_cal = np.not_equal(dists, -1)
        num_dist_cal = dist_cal.sum()
        if num_dist_cal > 0:
            return np.less(dists[dist_cal], self.threshold).sum() * 1.0 / num_dist_cal
        else:
            return -1

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.
        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        num_joints = preds[0].shape[1]

        for label, pred in zip(labels, preds):
            norm = 1.0
            h = pred.shape[2]
            w = pred.shape[3]
            if self.hm_type == 'gaussian':
                pred, _ = get_max_pred(pred)
                label, _ = get_max_pred(label)
                norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
                # norm = np.ones((pred.shape[0], 2)) * np.array([h, h]) / 10

            pred = pred.asnumpy()
            label = label.asnumpy()
            dists = self._calc_dists(pred, label, norm)

            acc = 0
            sum_acc = 0
            cnt = 0

            for i in range(num_joints):
                acc = self._dist_acc(dists[i])
                if acc >= 0:
                    sum_acc += acc
                    cnt += 1

            self.sum_metric += sum_acc
            self.num_inst += cnt


def get_max_pred(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = nd.argmax(heatmaps_reshaped, 2)
    maxvals = nd.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = nd.floor((preds[:, :, 1]) / width)

    pred_mask = nd.tile(nd.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
