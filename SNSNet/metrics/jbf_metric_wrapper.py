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
from mmpose.evaluation.metrics.functional import (oks_nms, soft_oks_nms, transform_ann, transform_pred,
                          transform_sigmas)

from .jbf_visualizer import JBFVisualizer

# from mmpose.visualization import JBFVisualizer
import numpy as np
import matplotlib.pyplot as plt

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
                    self.visualizer.visualize_jbf(f'{self.outfile_prefix}/vis/{self.count:03d}_{id}_{img_id}.png', img_path, \
                                                  mask_body, mask_joint, mask_joints, mask_flow, input_size, input_center, input_scale)
                    self.vis_flag = False

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if self.vis:
            self.vis_flag = True
            self.count += 1

        return self.metric.compute_metrics(results)