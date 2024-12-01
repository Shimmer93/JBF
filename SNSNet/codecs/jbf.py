from typing import Optional, Tuple, List

import numpy as np
from scipy.ndimage import gaussian_filter
from copy import deepcopy

from mmpose.registry import KEYPOINT_CODECS
from mmpose.datasets.datasets.body import str_to_dataset
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.codecs.base import BaseKeypointCodec
from mmpose.codecs.utils.gaussian_heatmap import (generate_gaussian_heatmaps,
                                     generate_unbiased_gaussian_heatmaps)
from mmpose.codecs.utils.post_processing import get_heatmap_maximum
from mmpose.codecs.utils.refinement import refine_keypoints, refine_keypoints_dark

def draw_line(canvas, start, end, value=1, overlength=0):
    # canvas: torch.Tensor of shape (height, width)
    # start: torch.Tensor of shape (2,)
    # end: torch.Tensor of shape (2,)

    # Bresenham's line algorithm
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    h, w = canvas.shape[0], canvas.shape[1]
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    if overlength > 0:
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            x0 = int(np.clip(x0 - dx * overlength, 0, h-1))
            y0 = int(np.clip(y0 - dy * overlength, 0, w-1))
            x1 = int(np.clip(x1 + dx * overlength, 0, h-1))
            y1 = int(np.clip(y1 + dy * overlength, 0, w-1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 >= 0 and x0 < w and y0 >= 0 and y0 < h:
        canvas[y0, x0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def skeleton_to_body_mask(keypoints, links, height, width):
    masks = np.zeros((keypoints.shape[0], height, width))
    for i in range(keypoints.shape[0]):
        for kp_pair in links:
            # if kp_pair[0] + kp_pair[1] == 10 or kp_pair[0] + kp_pair[1] == 8:
            #     continue
            start = keypoints[i, kp_pair[0], :2].astype(np.int32)
            end = keypoints[i, kp_pair[1], :2].astype(np.int32)
            if start[0] < 0 or start[0] >= width or start[1] < 0 or start[1] >= height or \
                end[0] < 0 or end[0] >= width or end[1] < 0 or end[1] >= height:
                continue
            draw_line(masks[i], start, end, value=1, overlength=0)
        masks[i] = gaussian_filter(masks[i], sigma=3, radius=1)
    mask = np.sum(masks, axis=0)
    return mask

def skeleton_to_joint_mask(keypoints, height, width):
    masks = np.zeros((keypoints.shape[0], keypoints.shape[1], height, width))
    for i in range(keypoints.shape[0]):
        for j, pt in enumerate(keypoints[i]):
            x, y = int(pt[0]), int(pt[1])
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            masks[i, j, y, x] = 1
            masks[i, j] = gaussian_filter(masks[i, j], sigma=3, radius=1)
    masks = np.sum(masks, axis=0)
    return masks

def get_keypoint_weights(keypoints, keypoints_visible, height, width):
    keypoint_weights = deepcopy(keypoints_visible)
    for i in range(keypoints.shape[0]):
        for j, pt in enumerate(keypoints[i]):
            if pt[0] < 0 or pt[0] >= width or pt[1] < 0 or pt[1] >= height:
                keypoint_weights[i, j] = 0
            if keypoints_visible[i, j] < 0.5:
                keypoint_weights[i, j] = 0
    return keypoint_weights

def get_dataset_links(dataset_type):
    links = list(str_to_dataset[dataset_type]._load_metainfo()["skeleton_links"])
    if 'Coco' in dataset_type:
        links.remove((5, 11))
        links.remove((6, 12))
        links.append((5, 12))
        links.append((6, 11))
    return links
        

def get_dataset_joint_groups(dataset_type):
    if 'Jhmdb' in dataset_type:
        return [[0], [1], [2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]] 

@KEYPOINT_CODECS.register_module()
class JBFCodec(BaseKeypointCodec):

    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(masks='masks', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 mask_size: Tuple[int, int],
                 sigma: float,
                 dataset_type: str,
                 unbiased: bool = False,
                 blur_kernel_size: int = 3,
                 use_flow: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.mask_size = mask_size
        self.sigma = sigma
        self.unbiased = unbiased

        self.blur_kernel_size = blur_kernel_size
        self.links = get_dataset_links(dataset_type)

        self.use_flow = use_flow

        self.scale_factor = (np.array(input_size) /
                             mask_size).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        w, h = self.mask_size
        w_, h_ = self.input_size
        body_mask = skeleton_to_body_mask(keypoints / self.scale_factor, self.links, h, w)
        joint_mask = skeleton_to_joint_mask(keypoints / self.scale_factor, h, w)
        keypoint_weights = get_keypoint_weights(keypoints, keypoints_visible, h_, w_)

        masks = np.concatenate([np.expand_dims(body_mask, axis=0), joint_mask], axis=0)

        encoded = dict(
            masks=masks,
            keypoint_weights=keypoint_weights,
        )

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        masks = encoded.copy()
        K, H, W = masks.shape

        masks_joints = masks[1:-1,...] if self.use_flow else masks[1:,...]

        keypoints, scores = get_heatmap_maximum(masks_joints)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        if self.unbiased:
            # Alleviate biased coordinate
            keypoints = refine_keypoints_dark(
                keypoints, masks_joints, blur_kernel_size=self.blur_kernel_size)

        else:
            keypoints = refine_keypoints(keypoints, masks_joints)

        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores