import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

import os
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MessageHub, MMLogger, print_log
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS
from mmpose.structures.bbox import bbox_xyxy2xywh, get_warp_matrix
from ..functional import (oks_nms, soft_oks_nms, transform_ann, transform_pred,
                          transform_sigmas)

# from mmpose.visualization import JBFVisualizer
import numpy as np
import matplotlib.pyplot as plt

class JBFVisualizer:

    def get_predefined_colors(self, num_kps):
        color_list = \
            [[ 34, 74, 243], [197, 105, 1], [23, 129, 240], [188, 126, 228], [115, 121, 232], [142, 144, 20], 
            [126, 250, 110], [217, 132, 212], [81, 191, 65], [103, 227, 95], [163, 179, 130], [120, 102, 117],
            [199, 85, 111], [98, 251, 87], [59, 24, 47], [55, 244, 124], [251, 221, 136], [186, 25, 19], 
            [172, 81, 95], [96, 76, 118], [11, 43, 76], [181, 55, 80], [157, 186, 192], [80, 185, 205],
            [12, 94, 115], [30, 220, 233], [144, 67, 163], [125, 159, 138], [136, 210, 185], [235, 25, 213]]
        
        colors = np.array(color_list[:num_kps])
        return colors

    def mask_to_image(self, mask, num_kps, convert_to_bgr=False):
        """
        Expects a two dimensional mask image of shape.

        Args:
            mask (np.ndarray): Mask image of shape [H,W]
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
            np.ndarray: Mask visualization image of shape [H,W,3]
        """
        assert mask.ndim == 2, 'input mask must have two dimensions'
        canvas = np.ones((mask.shape[0], mask.shape[1], 3), np.uint8) * 255
        colors = self.get_predefined_colors(num_kps)
        for i in range(num_kps):
            canvas[mask == i+1] = colors[i]
        if convert_to_bgr:
            canvas = canvas[...,::-1]
        return canvas

    def mask_to_joint_images(self, masks, num_kps, convert_to_bgr=False):
        assert masks.ndim == 3, 'input mask must have three dimensions'
        assert masks.shape[0] == num_kps, 'input mask must have shape [num_kps, H, W]'
        canvas = np.ones((num_kps, masks.shape[1], masks.shape[2], 3), np.uint8) * 255
        colors = self.get_predefined_colors(num_kps)
        for i in range(num_kps):
            canvas[i, masks[i] == 1] = colors[i]
        if convert_to_bgr:
            canvas = canvas[...,::-1]
        return canvas

    def visualize_psm(self, save_path, img_path, mask_body, mask_joint, mask_joints, mask_flow=None, input_size=None, input_center=None, input_scale=None, fig_h=10, fig_w=10, nrows=4, ncols=5):
        num_kps = mask_joints.shape[0]
        if mask_flow is not None:
            assert nrows * ncols >= num_kps + 4, f'{nrows} * {ncols} < {num_kps} + 4'
        else:
            assert nrows * ncols >= num_kps + 3, f'{nrows} * {ncols} < {num_kps} + 3'

        # for i in range(len(mask_body)):
        img = plt.imread(img_path)
        if input_size is None:
            warped_img = img
        else:
            warp_matrix = get_warp_matrix(input_center, input_scale, 0, input_size)
            warped_img = cv2.warpAffine(img, warp_matrix, input_size, flags=cv2.INTER_LINEAR)
        mask_body = self.mask_to_image(mask_body, 1)
        mask_joint = self.mask_to_image(mask_joint, num_kps)
        mask_joints = self.mask_to_joint_images(mask_joints, num_kps)
        data = [warped_img, mask_body, mask_joint]

        if mask_flow is not None:
            mask_flow = self.mask_to_image(mask_flow, 1)
            data.append(mask_flow)
        data += [mask_joints[j] for j in range(num_kps)]

        fig = plt.figure(figsize=(fig_h, fig_w))
        for j, d in enumerate(data):
            ax = fig.add_subplot(nrows, ncols, j+1)
            ax.imshow(d)
        fig.savefig(save_path)
        fig.clear()

        plt.close(fig)

def write_psm(save_path, joint_masks, body_mask=None, obj_mask=None, rescale_ratio=1.0):

    os.makedirs(save_path.rsplit('/', 1)[0], exist_ok=True)

    out_masks = joint_masks
    if body_mask is not None:
        out_masks = np.concatenate([out_masks, np.expand_dims(body_mask, axis=0)], axis=0)
    if obj_mask is not None:
        out_masks = np.concatenate([out_masks, np.expand_dims(obj_mask, axis=0)], axis=0)
    # out_masks = F.interpolate(out_masks.unsqueeze(0), scale_factor=1.0/rescale_ratio, \
    #                           mode='bilinear', align_corners=False).squeeze(0)
    h, w = out_masks.shape[-2:]
    out_masks = np.stack([cv2.resize(mask, dsize=(int(w/rescale_ratio), int(h/rescale_ratio)), interpolation=cv2.INTER_LINEAR) for mask in out_masks])
    out_masks = (out_masks * 255).astype(np.uint8)
    J, H, W = out_masks.shape

    nw = 4
    nh = int(np.ceil(J / nw))
    canvas = np.zeros((H * nh, W * nw), dtype=np.uint8)
    for i in range(J):
        x = (i % nw) * W
        y = (i // nw) * H
        canvas[y:y+H, x:x+W] = out_masks[i]
    canvas[-1, -1] = J
    canvas[-1, -2] = H
    canvas[-1, -3] = W
    
    Image.fromarray(canvas).save(save_path)

@METRICS.register_module()
class JBFMetricWrapper(BaseMetric):
    def __init__(self, 
                 metric_config: Dict,
                 vis: bool = True,
                 save: bool = False,
                 use_flow: bool = False,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.metric_config = metric_config
        self.vis = vis
        self.vis_flag = vis
        self.count = 0
        self.save = save
        self.use_flow = use_flow

        self.metric = METRICS.build(metric_config)
        self.outfile_prefix = self.metric.outfile_prefix if outfile_prefix is None else outfile_prefix

        os.makedirs(self.outfile_prefix, exist_ok=True)
        os.makedirs(f'{self.outfile_prefix}/vis', exist_ok=True)
        os.makedirs(f'{self.outfile_prefix}/results', exist_ok=True)
        
        self.visualizer = JBFVisualizer()

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        self.metric.process(data_batch, data_samples)
        self.results.append(self.metric.results[-1])

        for data_sample in data_samples:
            if 'pred_fields' in data_sample:

                id = data_sample['id']
                img_id = data_sample['img_id']
                img_path = data_sample['img_path']
                category_id = data_sample.get('category_id', 1)
                masks = data_sample['pred_fields']['heatmaps'].detach().cpu()

                if 'input_size' in data_sample:
                    input_size = data_sample['input_size']
                    input_center = data_sample['input_center']
                    input_scale = data_sample['input_scale']
                else:
                    input_size = None
                    input_center = None
                    input_scale = None

                mask_body = (masks[0] > 0.5).float()
                mask_body = mask_body.numpy()
                mask_body_raw = masks[0].numpy()

                if self.use_flow:
                    mask_joints = (masks[1:-1] > 0.5).float()
                    mask_joints_raw = masks[1:-1].numpy()
                    mask_flow = (masks[-1] > 0.5).float()
                    mask_flow = mask_flow.numpy()
                    mask_flow_raw = masks[-1].numpy()
                else:
                    mask_joints = (masks[1:] > 0.5).float()
                    mask_joints_raw = masks[1:].numpy()
                    mask_flow = None
                    mask_flow_raw = None

                mask_joints_neg = (torch.max(mask_joints, dim=0, keepdim=True)[0] < 0.5).float()
                mask_joint = torch.argmax(torch.cat([mask_joints_neg, mask_joints], dim=0), dim=0)
                mask_joint = mask_joint.numpy()
                mask_joints = mask_joints.numpy()

                if self.vis_flag:
                    self.visualizer.visualize_psm(f'vis/{self.count:03d}_{id}_{img_id}.png', img_path, \
                                                  mask_body, mask_joint, mask_joints, mask_flow, input_size, input_center, input_scale)
                    self.vis_flag = False

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if self.vis:
            self.vis_flag = True
            self.count += 1

        return self.metric.compute_metrics(results)