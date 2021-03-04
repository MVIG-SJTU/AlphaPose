# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
import torch
import torch.nn as nn

from .builder import LOSS

from alphapose.utils.transforms import _integral_tensor


class IngetralCoordinate(torch.autograd.Function):
    ''' Symmetry integral regression function.
    '''
    AMPLITUDE = 2

    @staticmethod
    def forward(ctx, input):
        assert isinstance(
            input, torch.Tensor), 'IngetralCoordinate only takes input as torch.Tensor'
        input_size = input.size()
        weight = torch.arange(
            input_size[-1], dtype=input.dtype, layout=input.layout, device=input.device)
        ctx.input_size = input_size
        output = input.mul(weight)
        ctx.save_for_backward(weight, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, output = ctx.saved_tensors
        output_coord = output.sum(dim=2, keepdim=True)
        weight = weight[None, None, :].repeat(
            output_coord.shape[0], output_coord.shape[1], 1)
        weight_mask = torch.ones(weight.shape, dtype=grad_output.dtype,
                                 layout=grad_output.layout, device=grad_output.device)
        weight_mask[weight < output_coord] = -1
        weight_mask[output_coord.repeat(
            1, 1, weight.shape[-1]) > ctx.input_size[-1]] = 1
        weight_mask *= IngetralCoordinate.AMPLITUDE
        return grad_output.mul(weight_mask)


@LOSS.register_module
class L1JointRegression(nn.Module):
    ''' L1 Joint Regression Loss
    '''
    def __init__(self, OUTPUT_3D=False, size_average=True, reduce=True, NORM_TYPE='softmax'):
        super(L1JointRegression, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.output_3d = OUTPUT_3D
        self.norm_type = NORM_TYPE

        self.integral_operation = IngetralCoordinate.apply

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        if self.output_3d:
            num_joints = int(gt_joints_vis.shape[1] / 3)
        else:
            num_joints = int(gt_joints_vis.shape[1] / 2)
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // num_joints if self.output_3d else 1

        pred_jts, pred_scores = _integral_tensor(
            preds, num_joints, self.output_3d, hm_width, hm_height, hm_depth, integral_operation=self.integral_operation, norm_type=self.norm_type)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_l1_loss(pred_jts, pred_scores, gt_joints, gt_joints_vis, self.size_average)


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


def weighted_l1_loss(input, scores, target, weights, size_average):
    out = torch.abs(input - target)
    out = out * weights
    #out_of_scores = torch.abs(scores - torch.ones_like(scores))
    #out_of_scores = out_of_scores.reshape((out_of_scores.shape[0], -1))
    #out_of_scores = out_of_scores * weights[:, 0::2]
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()


LOSS.register_module(torch.nn.MSELoss)
