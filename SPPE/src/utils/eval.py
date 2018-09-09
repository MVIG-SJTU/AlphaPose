from opt import opt
try:
    from utils.img import transformBoxInvert, transformBoxInvert_batch
except ImportError:
    from SPPE.src.utils.img import transformBoxInvert, transformBoxInvert_batch
import torch


class DataLogger(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def accuracy(output, label, dataset):
    if type(output) == list:
        return accuracy(output[opt.nStack - 1], label[opt.nStack - 1], dataset)
    else:
        return heatmapAccuracy(output.cpu().data, label.cpu().data, dataset.accIdxs)


def heatmapAccuracy(output, label, idxs):
    preds = getPreds(output)
    gt = getPreds(label)

    norm = torch.ones(preds.size(0)) * opt.outputResH / 10
    dists = calc_dists(preds, gt, norm)
    #print(dists)
    acc = torch.zeros(len(idxs) + 1)
    avg_acc = 0
    cnt = 0
    for i in range(len(idxs)):
        acc[i + 1] = dist_acc(dists[idxs[i] - 1])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1
    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


def getPreds(hm):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert hm.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hm.view(hm.size(0), hm.size(1), -1), 2)

    maxval = maxval.view(hm.size(0), hm.size(1), 1)
    idx = idx.view(hm.size(0), hm.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hm.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hm.size(3))

    # pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    # preds *= pred_mask
    return preds


def calc_dists(preds, target, normalize):
    preds = preds.float().clone()
    target = target.float().clone()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 0 and target[n, c, 1] > 0:
                dists[c, n] = torch.dist(
                    preds[n, c, :], target[n, c, :]) / normalize[n]
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).float().sum() * 1.0 / dists.ne(-1).float().sum()
    else:
        return - 1


def postprocess(output):
    p = getPreds(output)

    for i in range(p.size(0)):
        for j in range(p.size(1)):
            hm = output[i][j]
            pX, pY = int(round(p[i][j][0])), int(round(p[i][j][1]))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
                diff = torch.Tensor((hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                p[i][j] += diff.sign() * 0.25
    p -= 0.5

    return p


def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    '''
    Get keypoint location from heatmaps
    '''

    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(round(float(preds[i][j][1])))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                preds[i][j] += diff.sign() * 0.25
    preds += 0.2

    preds_tf = torch.zeros(preds.size())

    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW)

    return preds, preds_tf, maxval


def getPrediction_batch(hms, pt1, pt2, inpH, inpW, resH, resW):
    '''
    Get keypoint location from heatmaps
    pt1, pt2:   [n, 2]
    OUTPUT:
        preds:  [n, 17, 2]
    '''

    assert hms.dim() == 4, 'Score maps should be 4-dim'
    flat_hms = hms.view(hms.size(0), hms.size(1), -1)
    maxval, idx = torch.max(flat_hms, 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    idx_up = (idx - hms.size(3)).clamp(0, flat_hms.size(2) - 1)
    idx_down = (idx + hms.size(3)).clamp(0, flat_hms.size(2) - 1)
    idx_left = (idx - 1).clamp(0, flat_hms.size(2) - 1)
    idx_right = (idx + 1).clamp(0, flat_hms.size(2) - 1)

    maxval_up = flat_hms.gather(2, idx_up)
    maxval_down = flat_hms.gather(2, idx_down)
    maxval_left = flat_hms.gather(2, idx_left)
    maxval_right = flat_hms.gather(2, idx_right)

    diff1 = (maxval_right - maxval_left).sign() * 0.25
    diff2 = (maxval_down - maxval_up).sign() * 0.25
    diff1[idx_up <= hms.size(3)] = 0
    diff1[idx_down / hms.size(3) >= (hms.size(3) - 1)] = 0
    diff2[(idx_left % hms.size(3)) == 0] = 0
    diff2[(idx_left % hms.size(3)) == (hms.size(3) - 1)] = 0

    preds[:, :, 0] += diff1.squeeze(-1)
    preds[:, :, 1] += diff2.squeeze(-1)

    preds_tf = torch.zeros(preds.size())
    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW)

    return preds, preds_tf, maxval
