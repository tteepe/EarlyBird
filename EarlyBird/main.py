import os.path as osp

import torch
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import json
from pytorch_metric_learning.losses import SupConLoss
from evaluation.mAP_nuscenes import NuscenesDetectionEvaluator
from models import MVDet
from models.loss import FocalLoss, compute_rot_loss
from tracking.multitracker import JDETracker
from utils import vox, basic, decode
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics_pedestrian
from nuscenes.eval.common.config import config_factory

class WorldTrackModel(pl.LightningModule):
    def __init__(
            self,
            model_name='mvdet',
            encoder_name='res18',
            learning_rate=1e-3,
            resolution=(200, 4, 200),
            bounds=(-75, 75, -75, 75, -1, 5),
            num_cameras=None,
            num_ids=None,
            depth=(100, 2.0, 25),
            scene_centroid=(0.0, 0.0, 0.0),
            max_detections=60,
            conf_threshold=0.4,
            gating_threshold=1000,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.Y, self.Z, self.X = self.resolution
        self.bounds = bounds
        self.max_detections = max_detections
        self.D, self.DMIN, self.DMAX = depth
        self.conf_threshold = conf_threshold

        # Loss
        self.center_loss_fn = FocalLoss()
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.contrastive_loss = SupConLoss()

        # Test
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        self.mota_frame_gt_list, self.mota_frame_pred_list = [], []
        self.mAP_gt_dict, self.mAP_pred_dict = {}, {}
        self.mAP_results_gt, self.mAP_results_pred = {}, {}
        self.test_tracker = JDETracker(conf_thres=self.conf_threshold, gating_threshold=gating_threshold)

        # Model
        if model_name == 'mvdet':
            self.model = MVDet(self.Y, self.Z, self.X, encoder_type=self.encoder_name,
                               num_cameras=num_cameras, num_ids=num_ids)
        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        self.id_head = torch.nn.Linear(self.model.decoder.reid_feat, num_ids)
        self.scene_centroid = torch.tensor(scene_centroid, device=self.device).reshape([1, 3])
        self.vox_util = vox.VoxelUtil(self.Y, self.Z, self.X, scene_centroid=self.scene_centroid, bounds=self.bounds)
        self.save_hyperparameters()

    def forward(self, item):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        ref_T_global: (B,4,4)
        vox_util: vox util object
        """
        return self.model(
            rgb_cams=item['img'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            vox_util=self.vox_util,
            ref_T_global=item['ref_T_global'],
        )

    def loss(self, target, output):
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']
        feat_bev_e = output['instance_id_feat']  # B,nids,Y,X
        center_img_e = output['img_center']
        offset_img_e = output['img_offset']
        size_img_e = output['img_size']
        feat_img_e = output['img_id_feat']

        valid_g = target['valid_bev']
        center_g = target['center_bev']
        offset_g = target['offset_bev']
        size_g = target['size_bev']
        rotbin_g = target['rotbin_bev']
        rotres_g = target['rotres_bev']
        # depth_g = target['depth']
        id_g = target['pid_bev']  # B,1,Y,X

        B, S = target['center_img'].shape[:2]
        center_img_g = basic.pack_seqdim(target['center_img'], B)
        offset_img_g = basic.pack_seqdim(target['offset_img'], B)
        size_img_g = basic.pack_seqdim(target['size_img'], B)
        valid_img_g = basic.pack_seqdim(target['valid_img'], B)
        id_img_g = basic.pack_seqdim(target['pid_img'], B)  # B*S,1,H/8,W/8

        center_loss = self.center_loss_fn(basic._sigmoid(center_e), center_g)
        offset_loss = torch.abs(offset_e - offset_g).sum(dim=1, keepdim=True)
        offset_loss = 10 * basic.reduce_masked_mean(offset_loss, valid_g)

        if size_g.any():
            size_loss = torch.abs(size_e - size_g).sum(dim=1, keepdim=True)
            size_loss = basic.reduce_masked_mean(size_loss, valid_g)
            rot_loss = compute_rot_loss(rot_e, rotbin_g, rotres_g, valid_g)
        else:
            size_loss = torch.tensor(0.)
            rot_loss = torch.tensor(0.)

        center_factor = 1 / (2 * torch.exp(self.model.center_weight))
        center_loss = 10.0 * center_factor * center_loss
        center_uncertainty_loss = 0.5 * self.model.center_weight

        offset_factor = 1 / (2 * torch.exp(self.model.offset_weight))
        offset_loss = offset_factor * offset_loss
        offset_uncertainty_loss = 0.5 * self.model.offset_weight

        size_factor = 1 / (2 * torch.exp(self.model.size_weight))
        size_loss = size_factor * size_loss
        size_uncertainty_loss = 0.5 * self.model.size_weight

        rot_factor = 1 / (2 * torch.exp(self.model.rot_weight))
        rot_loss = rot_factor * rot_loss
        rot_uncertainty_loss = 0.5 * self.model.rot_weight

        # img loss
        center_img_loss = self.center_loss_fn(basic._sigmoid(center_img_e), center_img_g) / S
        offset_img_loss = basic.reduce_masked_mean(
            torch.abs(offset_img_e - offset_img_g).sum(dim=1, keepdim=True), valid_img_g) / S
        size_img_loss = basic.reduce_masked_mean(
            torch.abs(size_img_e - size_img_g).sum(dim=1, keepdim=True), valid_img_g) / (10 * S)

        # re_id loss
        valid_g = valid_g.flatten(1)
        valid_img_g = valid_img_g.flatten(1)
        targets = torch.cat([
            id_g.flatten(2).transpose(1, 2)[valid_g],
            id_img_g.flatten(2).transpose(1, 2)[valid_img_g]
        ]).squeeze(-1)
        feats = torch.cat([
            feat_bev_e.flatten(2).transpose(1, 2)[valid_g],
            feat_img_e.flatten(2).transpose(1, 2)[valid_img_g]
        ])
        ids = self.id_head(feats)
        reid_class_loss = self.classification_loss(ids, targets)
        reid_contras_loss = self.contrastive_loss(feats, targets)

        loss_dict = {
            'center_loss': center_loss,
            'offset_loss': offset_loss,
            'size_loss': size_loss,
            'rot_loss': rot_loss,

            'center_img': center_img_loss,
            'offset_img': offset_img_loss,
            'size_img': size_img_loss,

            'reid_class_loss': reid_class_loss,
            'reid_contras_loss': reid_contras_loss,
        }
        stats_dict = {
            'center_uncertainty_loss': center_uncertainty_loss,
            'offset_uncertainty_loss': offset_uncertainty_loss,
            'size_uncertainty_loss': size_uncertainty_loss,
            'rot_uncertainty_loss': rot_uncertainty_loss,
        }
        total_loss = sum(loss_dict.values()) + sum(stats_dict.values())

        return total_loss, loss_dict, stats_dict

    def training_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        total_loss, loss_dict, stats_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)
        for key, value in stats_dict.items():
            self.log(f'stats/{key}', value, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        if batch_idx % 100 == 0:
            self.plot_data(target, output, batch_idx)

        total_loss, loss_dict, _ = self.loss(target, output)
        total_loss -= loss_dict['reid_class_loss']

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        ref_T_global = item['ref_T_global']
        global_T_ref = torch.inverse(ref_T_global)

        # output on bev plane
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']
        id_e = output['instance_id_feat']

        xy_e, scores_e, ids_e, sizes_e, _ = decode.decoder(
            basic._sigmoid(center_e), offset_e, size_e, id_e, rz_e=rot_e, K=self.max_detections
        )

        # moda & modp
        mem_xyz = torch.cat((xy_e, torch.zeros_like(scores_e)), dim=2)  # B,K,3 [x_mem, y_mem, 0]
        ref_xyz = self.vox_util.Mem2Ref(mem_xyz, self.Y, self.Z, self.X)
        for frame, grid_gt, xyz, score in zip(item['frame'], item['grid_gt'], ref_xyz, scores_e.squeeze(2)):
            frame = int(frame.item())
            valid = score > self.conf_threshold

            self.moda_gt_list.extend([[frame, x.item(), y.item()] for x, y, _ in grid_gt[grid_gt.sum(1) != 0]])
            self.moda_pred_list.extend([[frame, x.cpu(), y.cpu()] for x, y, _ in xyz[valid]])

        # mota
        bev_det = torch.cat((ref_xyz[..., :2], torch.ones_like(ref_xyz[..., :2]), scores_e), dim=2).cpu()
        for frame, grid_gt, bev_det, bev_ids_e in zip(item['frame'], item['grid_gt'], bev_det, ids_e.cpu().numpy()):
            frame = int(frame.item())
            output_stracks = self.test_tracker.update(bev_det, bev_ids_e)
            self.mota_gt_list.extend([[frame, i.item(), -1, -1, -1, -1, 1, x.item(),  y.item(), -1]
                                      for x, y, i in grid_gt[grid_gt.sum(1) != 0]])
            self.mota_pred_list.extend([[frame, s.track_id, -1, -1, -1, -1, s.score.item()] + s.tlwh.tolist()[:2] + [-1]
                                        for s in output_stracks])

        # 3D mAP
        global_xyz_pred = torch.matmul(global_T_ref[0, :3, :3], ref_xyz.transpose(1, 2)).transpose(1, 2)
        global_xyz_gt = torch.matmul(global_T_ref[0, :3, :3], item['grid_gt'].float().transpose(1, 2)).transpose(1, 2)
        global_xyz_gt[::, 2] = item['grid_gt'][::, 2]

        for grid_gt, frame, xyz_pred, score in zip(global_xyz_gt, item['frame'], global_xyz_pred, scores_e.squeeze(2)):
            frame = int(frame.item())
            self.mAP_results_pred[str(frame)] = []
            self.mAP_results_gt[str(frame)] = []
            for (x_pred, y_pred, _), scr in zip(xyz_pred, score):
                sample_result_pred = {'sample_token': str(frame), 'translation': [x_pred.item(), y_pred.item(), 90],
                                      'size': [50, 50, 180], 'rotation': [0, 0, 0, 0], 'velocity': [0, 0],
                                      'detection_name': 'pedestrian', 'detection_score': scr.item(),
                                      'attribute_name': ''}
                self.mAP_results_pred[str(frame)].append(sample_result_pred)
            for x_gt, y_gt, _ in grid_gt:
                if x_gt == 0 and y_gt == 0:
                    continue
                sample_result_gt = {'sample_token': str(frame), 'translation': [x_gt.item(), y_gt.item(), 90],
                                    'size': [50, 50, 180], 'rotation': [0, 0, 0, 0], 'velocity': [0, 0],
                                    'detection_name': 'pedestrian', 'detection_score': 1,
                                    'attribute_name': ''}
                self.mAP_results_gt[str(frame)].append(sample_result_gt)

    def on_test_epoch_end(self):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'
        # moda & modp
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        gt_path = osp.join(log_dir, 'moda_gt.txt')
        np.savetxt(pred_path, np.array(self.moda_pred_list), '%f')
        np.savetxt(gt_path, np.array(self.moda_gt_list), '%d')
        recall, precision, moda, modp = modMetricsCalculator(osp.abspath(pred_path), osp.abspath(gt_path))
        self.log(f'detect/recall', recall)
        self.log(f'detect/precision', precision)
        self.log(f'detect/moda', moda)
        self.log(f'detect/modp', modp)

        # mota
        pred_path = osp.join(log_dir, 'mota_pred.txt')
        gt_path = osp.join(log_dir, 'mota_gt.txt')
        np.savetxt(pred_path, np.array(self.mota_pred_list), '%f', delimiter=',')
        np.savetxt(gt_path, np.array(self.mota_gt_list), '%f', delimiter=',')
        summary = mot_metrics_pedestrian(osp.abspath(pred_path), osp.abspath(gt_path))
        for key, value in summary.iloc[0].to_dict().items():
            self.log(f'track/{key}', value)

        # 3D mAP
        meta = {'use_camera': True, 'use_lidar': False, 'use_radar': False, 'use_map': False, 'use_external': False}
        self.mAP_pred_dict['meta'] = meta
        self.mAP_pred_dict['results'] = self.mAP_results_pred
        self.mAP_gt_dict['meta'] = meta
        self.mAP_gt_dict['results'] = self.mAP_results_gt
        pred_path = osp.join(log_dir, 'mAP_pred.json')
        gt_path = osp.join(log_dir, 'mAP_gt.json')
        with open(pred_path, 'w') as json_file:
            json.dump(self.mAP_pred_dict, json_file, indent=4)
        with open(gt_path, 'w') as json_file:
            json.dump(self.mAP_gt_dict, json_file, indent=4)
        nusc_eval = NuscenesDetectionEvaluator(config=config_factory('detection_cvpr_2019'),
                                               result_path=pred_path,
                                               gt_path=gt_path,
                                               verbose=False)
        nusc_metrics, nusc_metric_data_list = nusc_eval.evaluate()
        self.log(f'detect/mAP_3D', nusc_metrics.serialize()['mean_dist_aps']['pedestrian'] * 100)

    def plot_data(self, target, output, batch_idx=0):
        center_e = output['instance_center']
        center_g = target['center_bev']

        # save plots to tensorboard in eval loop
        writer = self.logger.experiment
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(center_g[0].sigmoid().squeeze().cpu().numpy(), cmap='hot', interpolation='nearest')
        ax2.imshow(center_e[0].sigmoid().squeeze().cpu().numpy(), cmap='hot', interpolation='nearest')
        ax1.set_title('center_g')
        ax2.set_title('center_e')
        plt.tight_layout()
        writer.add_figure(f'plot/{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }


if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("model.bounds", "data.init_args.bounds")


    cli = MyLightningCLI(WorldTrackModel)
