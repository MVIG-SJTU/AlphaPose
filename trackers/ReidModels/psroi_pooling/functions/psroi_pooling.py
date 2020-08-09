import torch
from torch.autograd import Function
from .._ext import psroi_pooling


class PSRoIPoolingFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

        self.output = None
        self.mappingchannel = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new().resize_(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_()
        mappingchannel = torch.IntTensor(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_().cuda(features.get_device())

        rtn = psroi_pooling.psroi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                 self.group_size, self.output_dim,
                                                 features, rois, output, mappingchannel)
        assert rtn > 0
        self.output = output
        self.mappingchannel = mappingchannel
        self.rois = rois
        self.feature_size = features.size()
        # print features.max(), features.min()
        # print rois.max(), rois.min()
        # print output.max(), output.min()
        return output

    def backward(self, grad_output):
        assert (self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()

        psroi_pooling.psroi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                  self.output_dim,
                                                  grad_output, self.rois, grad_input, self.mappingchannel)
        return grad_input, None
