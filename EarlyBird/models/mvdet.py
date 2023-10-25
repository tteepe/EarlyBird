import torch
import torch.nn as nn

import utils.geom
import utils.vox
import utils.basic

from kornia.geometry.transform.imgwarp import warp_perspective
from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18, Encoder_eff, Encoder_swin_t, Encoder_res34
from models.decoder import Decoder


class MVDet(nn.Module):
    def __init__(self, Y, Z, X,
                 rand_flip=False,
                 num_cameras=None,
                 num_ids=None,
                 latent_dim=512,
                 encoder_type='res18',
                 device=torch.device('cuda')):
        super().__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'res34', 'effb0', 'effb4', 'swin_t'])

        self.Y, self.Z, self.X = Y, Z, X
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.num_cameras = num_cameras

        self.mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

        # Encoder
        self.feat2d_dim = 128
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(self.feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(self.feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(self.feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(self.feat2d_dim)
        elif encoder_type == 'res34':
            self.encoder = Encoder_res34(self.feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(self.feat2d_dim)
        else:
            self.encoder = Encoder_eff(self.feat2d_dim, version='b4')

        if self.num_cameras is None:
            self.world_conv = nn.Sequential(
                nn.Conv3d(self.feat2d_dim, self.feat2d_dim, 3, padding=1),
                nn.InstanceNorm3d(latent_dim), nn.ReLU(),
                # nn.Conv3d(latent_dim, latent_dim, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            )
        else:
            self.world_feat = nn.Sequential(
                nn.Conv2d(self.feat2d_dim * self.num_cameras, latent_dim, kernel_size=3, padding=1),
                nn.InstanceNorm3d(latent_dim), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )

        # self.cam_out = nn.Sequential(
        #     nn.Conv2d(latent_dim, self.feat2d_dim, 3, padding=1),
        #     nn.InstanceNorm2d(self.feat2d_dim), nn.ReLU(),
        #     nn.Conv2d(self.feat2d_dim, self.feat2d_dim, 1, padding=0),
        # )

        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=2,
            n_ids=num_ids,
        )

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        """
        B, S, C, H, W = rgb_cams.shape
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams_ = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams_)  # B*S,4,4

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)  # B*S,3,H,W
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,latent_dim,H/8,W/8
        _, C, Hf, Wf = feat_cams_.shape
        sy = Hf / float(H)
        sx = Wf / float(W)
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)  # B*S,4,4

        featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cams_T_ref_[:, :3, [0, 1, 3]])  # B*S,3,3
        ref_T_mem = vox_util.get_ref_T_mem(B, self.Y, self.Z, self.X)  # B,4,4
        ref_T_mem = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3,3
        featpix_T_mem_ = torch.matmul(featpix_T_ref_, ref_T_mem)  # B*S,3,3
        mem_T_featpix = torch.inverse(featpix_T_mem_)  # B*S,3,3
        proj_mats = mem_T_featpix  # B*S,3,3

        # B*S,latent_dim,Y,X
        world_features_ = warp_perspective(feat_cams_, proj_mats, (self.Y, self.X), align_corners=False)

        world_features = __u(world_features_)  # B,S,latent_dim,Y,X
        if self.num_cameras is None:
            world_features = self.world_conv(world_features.permute(0, 2, 1, 3, 4))
            world_features = self.world_feat(world_features.sum(2, keepdim=True))
        else:
            world_features = self.world_feat(world_features.view(B, S * self.feat2d_dim, self.Y, self.X))

        # back_proj_mats = torch.inverse(proj_mats)  # B*S,3,3
        # world_features_cam = __p(world_features.unsqueeze(1).repeat(1, S, 1, 1, 1))
        # feat_cams_back_ = warp_perspective(world_features_cam, back_proj_mats, (Hf, Wf), align_corners=False)
        # feat_cams_ = feat_cams_ + self.cam_out(feat_cams_back_)

        out_dict = self.decoder(world_features, feat_cams_,
                                (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        return out_dict
