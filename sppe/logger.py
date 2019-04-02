import cv2
import mxnet as mx
import numpy as np
from mxnet import nd
from sppe.utils.img import ToImg, ToTensor


_skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def board_writing(writer, loss, acc, iterations, dataset='Train'):
    if loss is not None:
        writer.add_scalar(
            '{}/Loss'.format(dataset), loss, iterations)
    writer.add_scalar(
        '{}/Acc'.format(dataset), acc, iterations)


def visdom_writing(vis, outputs, labels, inputs, ori_inputs, win=None):
    if isinstance(labels, list):
        if len(outputs) > 1:
            outputs = nd.concat(*[out.as_in_context(mx.cpu())
                                  for out in outputs], dim=0)
            labels = nd.concat(*[label.as_in_context(mx.cpu()) for label in labels], dim=0)
            ori_inputs = nd.concat(*[ori_inp.as_in_context(mx.cpu()) for ori_inp in ori_inputs], dim=0)
        else:
            outputs = outputs[0].as_in_context(mx.cpu())
            labels = labels[0].as_in_context(mx.cpu())
            ori_inputs = ori_inputs[0].as_in_context(mx.cpu())

    _pic = ori_inputs[0].asnumpy()
    _label = labels[0].asnumpy()
    assert _pic.ndim == 3 and _label.ndim == 3
    _label_vis = _label[:, None, :, :]
    _pic_cv = ToImg(_pic)
    res_cor = []

    for i in range(17):
        max_val = np.max(_label[i])
        max_pos = np.where(_label[i] == max_val)
        x_pose = max_pos[1][0] * 4
        y_pose = max_pos[0][0] * 4
        res_cor.append((x_pose, y_pose))

    for pair in _skeleton:
        pair1, pair2 = pair
        p1 = res_cor[pair1 - 1]
        p2 = res_cor[pair2 - 1]
        if np.sum(p1) > 0 and np.sum(p2) > 0:
            cv2.line(_pic_cv, p1, p2, (255, 0, 0), 5)

    _pic = ToTensor(_pic_cv).asnumpy()

    win[1] = vis.images(_pic, win=win[1])
    win[2] = vis.images(_label_vis, win=win[2])

    return win


'''
if i % 10 == 0 and opt.visdom and False:
    _pic = inps[0][0].asnumpy()
    _label = labels[0].asnumpy()[:, np.newaxis, :, :]
    _out = out[0].asnumpy()[:, np.newaxis, :, :]
    _out[_out < 0] = 0
    global win
    win[1] = vis.images(_pic, win=win[1])
    win[2] = vis.images(_label, win=win[2])
    win[3] = vis.images(_out, win=win[3])

    #grads = [i.grad() for i in m.collect_params().values()]

    #assert len(grads) == len(param_names)
    # logging the gradients of parameters for checking convergence
'''
