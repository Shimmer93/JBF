# Copyright (c) OpenMMLab. All rights reserved.
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

    def visualize_jbf(self, save_path, img_path, mask_body, mask_joint, mask_joints, mask_flow=None, input_size=None, input_center=None, input_scale=None, fig_h=10, fig_w=10, nrows=4, ncols=5):
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