from collections import namedtuple
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.smpl.SMPL import SMPL_layer

ModelOutput = namedtuple(
    typename='ModelOutput',
    field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf',
                 'pred_uvd_jts', 'pred_xyz_jts_29', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
                 'pred_xyz_jts_17', 'pred_vertices', 'maxvals', 'cam_scale', 'cam_trans', 'cam_root',
                 'uvd_heatmap', 'transl', 'img_feat']
)
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


@SPPE.register_module
class Simple3DPoseBaseSMPLCam(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPLCam, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        backbone = ResNet

        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            ''' Load pretrained model '''
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(
            self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

        self.joint_pairs_29 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                               (22, 23), (25, 26), (27, 28))

        self.leaf_pairs = ((0, 1), (3, 4))
        self.root_idx_smpl = 0

        # mean shape
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())

        init_cam = torch.tensor([0.9, 0, 0])
        self.register_buffer(
            'init_cam',
            torch.Tensor(init_cam).float()) 
    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        self.focal_length = kwargs['FOCAL_LENGTH']
        self.input_size = 256.0

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

        # remap uv coordinate to input space
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * self.width_dim * 4
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * self.height_dim * 4
        # remap d to mm
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)

        dz = uvd_jts_new[:, :, 2]

        # transform in-bbox coordinate to image coordinate
        uv_homo_jts = torch.cat(
            (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
            dim=2)
        # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        # transform (u,v,1) to (x,y,z)
        cam_2d_homo = torch.cat(
            (uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
            dim=2)
        # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        # recover absolute z : (B,K) + (B,1)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        # multipy absolute z : (B,K,3) * (B,K,1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

        if return_relative:
            # (B,K,3) - (B,1,3)
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)

        xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)

        return xyz_jts

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        # flip
        if shift:
            pred_jts[:, :, 0] = - pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)

        return pred_jts
    
    def flip_xyz_coord(self, pred_jts, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        pred_jts[:, :, 0] = - pred_jts[:, :, 0]

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)

        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def forward(self, x, flip_item=None, flip_output=False, gt_uvd=None, gt_uvd_weight=None, **kwargs):
        
        batch_size = x.shape[0]

        # torch.cuda.synchronize()
        # model_start_t = time.time()

        x0 = self.preact(x)
        out = self.deconv_layers(x0)
        out = self.final_layer(out)

        # torch.cuda.synchronize()
        # preat_end_t = time.time()

        out = out.reshape((out.shape[0], self.num_joints, -1))

        out = norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape

        heatmaps = out / out.sum(dim=2, keepdim=True)
        maxvals, _ = torch.max(heatmaps, dim=2, keepdim=True)

        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))

        hm_x0 = heatmaps.sum((2, 3))
        hm_y0 = heatmaps.sum((2, 4))
        hm_z0 = heatmaps.sum((3, 4))

        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device)
        hm_x = hm_x0 * range_tensor
        hm_y = hm_y0 * range_tensor
        hm_z = hm_z0 * range_tensor

        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        #  -0.5 ~ 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        xc = x0

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        camScale = pred_camera[:, :1].unsqueeze(1)
        camTrans = pred_camera[:, 1:].unsqueeze(1)

        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)

        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone() # unit: 2.2m
        pred_xyz_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                                        * (pred_xyz_jts_29[:, :, 2:]*2.2 + camDepth) - camTrans # unit: m

        pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / 2.2 # unit: 2.2m

        camera_root = pred_xyz_jts_29[:, [0], ]*2.2
        camera_root[:, :, :2] += camTrans
        camera_root[:, :, [2]] += camDepth

        if not self.training:
            pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]

        if flip_item is not None:
            assert flip_output is not None
            pred_xyz_jts_29_orig, pred_phi_orig, pred_leaf_orig, pred_shape_orig = flip_item

        if flip_output:
            pred_xyz_jts_29 = self.flip_xyz_coord(pred_xyz_jts_29, flatten=False)
        if flip_output and flip_item is not None:
            pred_xyz_jts_29 = (pred_xyz_jts_29 + pred_xyz_jts_29_orig.reshape(batch_size, 29, 3)) / 2
        
        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        if flip_output:
            pred_phi = self.flip_phi(pred_phi)

        if flip_output and flip_item is not None:
            pred_phi = (pred_phi + pred_phi_orig) / 2
            pred_shape = (pred_shape + pred_shape_orig) / 2

        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * 2.2, # unit: meter
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        # pred_xyz_jts_24_struct = output.joints.float() / 2.2
        pred_xyz_jts_24_struct = output.joints.float() / 2
        #  -0.5 ~ 0.5
        # pred_xyz_jts_17 = output.joints_from_verts.float() / 2.2
        pred_xyz_jts_17 = output.joints_from_verts.float() / 2
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72) / 2
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        transl = pred_xyz_jts_29[:, 0, :] * 2.2 - pred_xyz_jts_17[:, 0, :] * 2.2
        transl[:, :2] += camTrans[:, 0]
        transl[:, 2] += camDepth[:, 0, 0]

        output = ModelOutput(
            pred_phi=pred_phi,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1),
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=maxvals,
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            # img_feat=x0
        )
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output
