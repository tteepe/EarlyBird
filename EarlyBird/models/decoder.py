import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.encoder import freeze_bn, UpsamplingConcat


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, n_ids):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        freeze_bn(backbone)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.reid_feat = 64
        self.feat2d = 128

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingConcat(256 + 128, 256)
        self.up2_skip = UpsamplingConcat(256 + 64, 256)
        self.up1_skip = UpsamplingConcat(256 + 512, shared_out_channels)

        # bev
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
        )
        self.instance_center_head[-1].bias.data.fill_(-2.19)

        self.instance_size_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 3, kernel_size=1, padding=0),
        )
        self.instance_rot_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 8, kernel_size=1, padding=0),
        )

        # img
        self.img_center_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, n_classes, kernel_size=1, padding=0),
        )
        self.img_offset_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, 2, kernel_size=1, padding=0),
        )
        self.img_size_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, 2, kernel_size=1, padding=0),
        )

        # re_id
        self.id_feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.reid_feat, kernel_size=1, padding=0),
        )
        self.img_id_feat_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, self.reid_feat, kernel_size=1, padding=0),
        )
        self.emb_scale = math.sqrt(2) * math.log(n_ids - 1)

    def forward(self, x, feat_cams, bev_flip_indices=None):
        b, c, h, w = x.shape

        # pad input
        m = 8
        ph, pw = math.ceil(h / m) * m - h, math.ceil(w / m) * m - w
        x = torch.nn.functional.pad(x, (ph, pw))

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        # Unpad
        x = x[..., ph // 2:h + ph // 2, pw // 2:w + pw // 2]

        # Extra upsample to (2xH, 2xW)
        # x = self.up_sample_2x(x)

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])  # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        # bev
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_size_output = self.instance_size_head(x)
        instance_rot_output = self.instance_rot_head(x)
        instance_id_feat_output = self.emb_scale * F.normalize(self.id_feat_head(x), dim=1)

        # img
        img_center_output = self.img_center_head(feat_cams)  # B*S,1,H/8,W/8
        img_offset_output = self.img_offset_head(feat_cams)  # B*S,2,H/8,W/8
        img_size_output = self.img_size_head(feat_cams)  # B*S,2,H/8,W/8
        img_id_feat_output = self.emb_scale * F.normalize(self.img_id_feat_head(feat_cams), dim=1)  # B*S,C,H/8,W/8

        return {
            # bev
            'raw_feat': x,
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_size': instance_size_output.view(b, *instance_size_output.shape[1:]),
            'instance_rot': instance_rot_output.view(b, *instance_rot_output.shape[1:]),
            'instance_id_feat': instance_id_feat_output.view(b, *instance_id_feat_output.shape[1:]),
            # img
            'img_center': img_center_output,
            'img_offset': img_offset_output,
            'img_size': img_size_output,
            'img_id_feat': img_id_feat_output,
        }
