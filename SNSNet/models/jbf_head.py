
from typing import Optional, Sequence, Tuple, Union
import time
import os

import torch
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn
import matplotlib.pyplot as plt

from mmpose.models.heads.base_head import BaseHead
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
OptIntSeq = Optional[Sequence[int]]

"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""

def get_min_max_from_masks(masks, padding=5):

    # masks: B H W
    B, H, W = masks.shape
    y_mins = []
    y_maxs = []
    x_mins = []
    x_maxs = []

    for i in range(B):
        y_nonzeros = torch.nonzero(torch.sum(masks[i], dim=1))
        x_nonzeros = torch.nonzero(torch.sum(masks[i], dim=0))
        y_mins.append(torch.min(y_nonzeros).item() if y_nonzeros.numel() > 0 else -1)
        y_maxs.append(torch.max(y_nonzeros).item() if y_nonzeros.numel() > 0 else -1)
        x_mins.append(torch.min(x_nonzeros).item() if x_nonzeros.numel() > 0 else -1)
        x_maxs.append(torch.max(x_nonzeros).item() if x_nonzeros.numel() > 0 else -1)
        
    y_mins = torch.tensor(y_mins, dtype=torch.int)
    y_maxs = torch.tensor(y_maxs, dtype=torch.int)
    x_mins = torch.tensor(x_mins, dtype=torch.int)
    x_maxs = torch.tensor(x_maxs, dtype=torch.int)

    return y_mins, y_maxs, x_mins, x_maxs

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def generate_regular_grid_point_coords(R, H, W, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        R (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (R, side_size^2, 2) that contains coordinates
            for the regular grids.
    """
    aff = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]], device=device)
    r = F.affine_grid(aff, torch.Size((1, 1, H, W)), align_corners=False)
    return r.view(1, -1, 2).expand(R, -1, -1)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


def point_sample_fine_grained_features(features, scale, point_coords):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.

    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.
        boxes (list[Boxes]): A list of I Boxes  objects that contain R_1 + ... + R_I = R boxes all
            together.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """

    image_shape = features.shape[-2:]
    image_shape = [image_shape[1] / scale - 1, image_shape[0] / scale - 1]
    point_coords_wrt_image = get_point_coords_wrt_image(image_shape, point_coords)

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(point_coords_wrt_image):
        point_features_per_image = []
        
        # for idx_feature, feature_map in enumerate(features_list):
        scale2 = torch.as_tensor(image_shape)
        point_coords_scaled = point_coords_wrt_image_per_image / scale2.to(features.device)
        point_features_per_image.append(
            point_sample(
                features[idx_img].unsqueeze(0),
                point_coords_scaled.unsqueeze(0),
                align_corners=False,
            )
            .squeeze(0)
        )
        # print(point_features_per_image[0].shape)
        point_features.append(torch.cat(point_features_per_image, dim=1))

    return torch.stack(point_features, dim=0), point_coords_wrt_image


def get_point_coords_wrt_image(image_shape, point_coords):
    """
    Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.

    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_image = point_coords.clone()
        point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
            image_shape[-2] - 1
        )
        point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
            image_shape[-1] - 1
        )
    return point_coords_wrt_image

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, activation=None):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                     groups, bias)
        self.activation = activation

    def forward(self, x):
        x = super(Conv2d, self).forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted or ground truth class
            for eash predicted mask.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = torch.topk(logits, k=2, dim=1)[0]
        gt_class_logits = gt_class_logits[:, 0:1, ...] - gt_class_logits[:, 1:2, ...]
    return -(torch.abs(gt_class_logits))

class SimplifiedPointRend(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained features and instance-wise MLP parameters as its input.
    """

    def __init__(self, in_channels, num_layers, channels, image_feature_enabled, positional_encoding_enabled, num_classes=1):
        """
        The following attributes are parsed from config:
            channels: the output dimension of each FC layers
            num_layers: the number of FC layers (including the final prediction layer)
            image_feature_enabled: if True, fine-grained image-level features are used
            positional_encoding_enabled: if True, positional encoding is used
        """
        super(SimplifiedPointRend, self).__init__()
        # fmt: off
        self.num_layers                         = num_layers + 1
        self.channels                           = channels
        self.image_feature_enabled              = image_feature_enabled
        self.positional_encoding_enabled        = positional_encoding_enabled
        self.num_classes                        = num_classes
        self.in_channels                        = in_channels
        # fmt: on

        if not self.image_feature_enabled:
            self.in_channels = 0
        if self.positional_encoding_enabled:
            self.in_channels += self.channels
            self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, self.channels//2)))

        assert self.in_channels > 0

        mlp_layers = []
        for l in range(self.num_layers):
            if l == 0:
                mlp_layers.append(nn.Linear(self.in_channels, self.channels))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(0.1))
            elif l == self.num_layers - 1:
                mlp_layers.append(nn.Linear(self.channels, self.num_classes))
            else:
                mlp_layers.append(nn.Linear(self.channels, self.channels))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(0.1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, fine_grained_features, point_coords):
        # features: [R, channels, K]
        # point_coords: [R, K, 2]
        num_instances = fine_grained_features.size(0)
        num_points = fine_grained_features.size(2)

        if num_instances == 0:
            return torch.zeros((0, 1, num_points), device=fine_grained_features.device)

        if self.positional_encoding_enabled:
            # locations: [R*K, 2]
            locations = 2 * point_coords.reshape(num_instances * num_points, 2) - 1
            locations = locations @ self.positional_encoding_gaussian_matrix.to(locations.device)
            locations = 2 * np.pi * locations
            locations = torch.cat([torch.sin(locations), torch.cos(locations)], dim=1)
            # locations: [R, C, K]
            locations = locations.reshape(num_instances, num_points, self.channels).permute(0, 2, 1)
            if not self.image_feature_enabled:
                fine_grained_features = locations
            else:
                fine_grained_features = torch.cat([locations, fine_grained_features], dim=1)

        # features [R, C, K]
        mask_feat = fine_grained_features.reshape(num_instances, self.in_channels, num_points)

        B, C, L = mask_feat.shape
        mask_feat = mask_feat.transpose(1, 2).reshape(B*L, C)
        point_logits = self.mlp(mask_feat)
        point_logits = point_logits.reshape(B, L, self.num_classes).transpose(1, 2)

        return point_logits

class SimplifiedPointRendMaskHead(nn.Module):
    def __init__(self, in_channels, train_num_points, subdivision_steps, scale, \
                 num_layers, channels, image_feature_enabled, positional_encoding_enabled, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        # fmt: off
        self.mask_point_on = True  # always on
        self.mask_point_train_num_points        = train_num_points
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps       = subdivision_steps
        # fmt: on
        self.scale = scale #1.0 / 4
        self.feat_scale = int((2 ** self.mask_point_subdivision_steps) * self.scale)

        self.point_head = SimplifiedPointRend(in_channels, num_layers, channels, image_feature_enabled, \
                                            positional_encoding_enabled, self.num_classes)

    def forward(self, features):
        """
        Args:
            features: B C H W
        """
        pass

    def _get_point_logits(self, fine_grained_features, point_coords):
        return self.point_head(fine_grained_features, point_coords)
    
    def _point_pooler(self, features, point_coords):
        # sample image-level features
        point_fine_grained_features, _ = point_sample_fine_grained_features(
            features, self.scale, point_coords
        )
        return point_fine_grained_features

    def _subdivision_inference(self, features):
        assert not self.training

        mask_point_subdivision_num_points = features.shape[-2] * features.shape[-1] // (self.feat_scale ** 2)
        mask_logits = None
        # +1 here to include an initial step to generate the coarsest mask
        # prediction with init_resolution, when mask_logits is None.
        # We compute initial mask by sampling on a regular grid. coarse_mask
        # can be used as initial mask as well, but it's typically very low-res
        # so it will be completely overwritten during subdivision anyway.
        for i in range(self.mask_point_subdivision_steps + 1):
            if mask_logits is None:
                # print(features.shape[-2]//self.feat_scale, features.shape[-1]//self.feat_scale)
                point_coords = generate_regular_grid_point_coords(
                    features.shape[0],
                    features.shape[-2]//self.feat_scale,
                    features.shape[-1]//self.feat_scale,
                    features.device,
                )
            else:
                mask_logits = F.interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                uncertainty_map = calculate_uncertainty(mask_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, mask_point_subdivision_num_points
                )

            # Run the point head for every point in point_coords
            fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                fine_grained_features, point_coords
            )

            if mask_logits is None:
                # Create initial mask_logits using point_logits on this regular grid
                R, C, _ = point_logits.shape
                mask_logits = point_logits.reshape(
                    R,
                    C,
                    features.shape[-2]//self.feat_scale,
                    features.shape[-1]//self.feat_scale,
                )
            else:
                # Put point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )
        return mask_logits

class BodyMaskHead(SimplifiedPointRendMaskHead):
    def __init__(self, in_channels, train_num_points, subdivision_steps, scale, \
                 num_layers, channels, image_feature_enabled, positional_encoding_enabled, num_classes=1):
        self.flag = True
        self.num_classes = num_classes
        super().__init__(in_channels, train_num_points, subdivision_steps, scale, \
                         num_layers, channels, image_feature_enabled, positional_encoding_enabled)

    def forward(self, features, skls=None, gt_masks=None, mode='fit'):
        """
        Args:
            features: B C H W
        """
        if mode == 'fit':
            point_coords, point_labels = self._sample_train_points_with_skeleton(features, skls, gt_masks)
            point_fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords
            )

            return point_logits, point_labels
        else:
            return F.sigmoid(self._subdivision_inference(features))
    
    def _sample_train_points_with_skeleton(self, features, skls, gt_masks):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        y_mins, y_maxs, x_mins, x_maxs = get_min_max_from_masks(gt_masks, padding=10)

        point_coords = []
        point_labels = []
        for(i, (y_min, y_max, x_min, x_max)) in enumerate(zip(y_mins, y_maxs, x_mins, x_maxs)):

            neg_mask = torch.ones((h, w), dtype=torch.bool, device=features.device)
            if y_min >= 0:
                assert y_max >= 0 and x_min >= 0 and x_max >= 0
                neg_mask[y_min:y_max, x_min:x_max] = False
            if torch.sum(neg_mask) == 0:
                neg_mask[0, 0] = True
            neg_idxs = torch.nonzero(neg_mask).flip(1)

            pos_mask = (gt_masks[i] > 0)
            pos_idxs = torch.nonzero(pos_mask).flip(1)

            if len(pos_idxs) == 0:
                num_samples = self.mask_point_train_num_points
                while len(neg_idxs) < num_samples:
                    neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
                neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
                point_coords.append(neg_idxs)
                point_labels.append(torch.zeros(num_samples))

            else:
                num_samples = self.mask_point_train_num_points // 2
                while len(neg_idxs) < num_samples:
                    neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
                neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
                while len(pos_idxs) < num_samples:
                    pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
                pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples]
                point_coords.append(torch.cat([neg_idxs, pos_idxs], dim=0))
                point_labels.append(torch.cat([torch.zeros(num_samples), torch.ones(num_samples)], dim=0))

        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        return point_coords, point_labels
    
class JointMaskHead(SimplifiedPointRendMaskHead):
    def __init__(self, in_channels, train_num_points, subdivision_steps, scale, \
                 num_layers, channels, image_feature_enabled, positional_encoding_enabled, num_classes=1):
        super().__init__(in_channels, train_num_points, subdivision_steps, scale, \
                 num_layers, channels, image_feature_enabled, positional_encoding_enabled, num_classes)

    def forward(self, features, skls=None, gt_masks=None, mode='fit'):
        """
        Args:
            features: B C H W
        """
        if mode == 'fit':
            point_coords, point_labels = self._sample_train_points_with_skeleton(features, skls, gt_masks)
            point_fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords
            )
            return point_logits, point_labels
        else:
            return F.sigmoid(self._subdivision_inference(features))
        
    def _sample_train_points_with_skeleton2(self, features, skls, gt_masks):
        assert self.training

        B, J, _, _ = gt_masks.shape
        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        min_maxs = []
        for j in range(J):
            min_maxs.append(tuple(get_min_max_from_masks(gt_masks[:, j]), padding=10))

        point_coords = []
        point_labels = []

        for i in range(B):
            coords_i = []
            labels_i = []

            for j in range(J):
                y_min, y_max, x_min, x_max = min_maxs[j]

                neg_mask = torch.ones((h, w), dtype=torch.bool, device=features.device)
                if y_min[i] >= 0:
                    assert y_max[i] >= 0 and x_min[i] >= 0 and x_max[i] >= 0
                    neg_mask[y_min[i]:y_max[i], x_min[i]:x_max[i]] = False
                if torch.sum(neg_mask) == 0:
                    neg_mask[0, 0] = True
                neg_idxs = torch.nonzero(neg_mask).flip(1)

                pos_mask = (gt_masks[i][j] > 0)
                pos_idxs = torch.nonzero(pos_mask).flip(1)

                if len(pos_idxs) == 0:
                    num_samples = self.mask_point_train_num_points
                    while len(neg_idxs) < num_samples:
                        neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
                    neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
                    coords_i.append(neg_idxs)
                    labels_i.append(torch.zeros(num_samples))

                else:
                    num_samples_neg = self.mask_point_train_num_points // 8 * 7
                    num_samples_pos = self.mask_point_train_num_points // 8
                    while len(neg_idxs) < num_samples_neg:
                        neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
                    neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples_neg]
                    while len(pos_idxs) < num_samples_pos:
                        pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
                    pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples_pos]
                    coords_i.append(torch.cat([neg_idxs, pos_idxs], dim=0))
                    labels_i.append(torch.cat([torch.zeros(num_samples_neg), torch.ones(num_samples_pos)], dim=0))

            point_coords.append(torch.stack(coords_i, dim=0))
            point_labels.append(torch.stack(labels_i, dim=0))

        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        return point_coords, point_labels

    def _sample_train_points_with_skeleton(self, features, skls, gt_masks):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        y_mins, y_maxs, x_mins, x_maxs = get_min_max_from_masks(gt_masks.sum(dim=1), padding=10)

        point_coords = []
        point_labels = []
        for(i, (y_min, y_max, x_min, x_max)) in enumerate(zip(y_mins, y_maxs, x_mins, x_maxs)):
            coords_i = []
            labels_i = []

            num_samples_total = int(self.mask_point_train_num_points * (gt_masks.shape[1]/8 + 1/2))

            num_samples_pos = self.mask_point_train_num_points // 8
            # pos_masks = []
            for j in range(gt_masks.shape[1]):
                pos_mask = (gt_masks[i][j] > 0)
                # pos_masks.append(pos_mask)
                pos_idxs = torch.nonzero(pos_mask).flip(1)
                if len(pos_idxs) == 0:
                    continue
                while len(pos_idxs) < num_samples_pos:
                    pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
                pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples_pos]
                coords_i.append(pos_idxs)
                labels_i.append(torch.ones(num_samples_pos) * (j+1))

            num_samples_pos_total = len(coords_i) * num_samples_pos

            num_samples_neg = num_samples_total - num_samples_pos_total
            neg_mask = torch.ones((h, w), dtype=torch.bool, device=features.device)
            if y_min >= 0:
                assert y_max >= 0 and x_min >= 0 and x_max >= 0
                neg_mask[y_min:y_max, x_min:x_max] = False
            if torch.sum(neg_mask) == 0:
                neg_mask[0, 0] = True
            neg_idxs = torch.nonzero(neg_mask).flip(1)

            while len(neg_idxs) < num_samples_neg:
                neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
            neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples_neg]
            coords_i.append(neg_idxs)
            labels_i.append(torch.zeros(num_samples_neg))

            point_coords.append(torch.cat(coords_i, dim=0))
            point_labels.append(torch.cat(labels_i, dim=0))

        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        return point_coords, point_labels
    
class FlowMaskHead(SimplifiedPointRendMaskHead):
    def __init__(self, in_channels, train_num_points, subdivision_steps, scale, \
                 num_layers, channels, image_feature_enabled, positional_encoding_enabled, num_classes=1):
        super().__init__(in_channels, train_num_points, subdivision_steps, scale, \
                 num_layers, channels, image_feature_enabled, positional_encoding_enabled, num_classes)

    def forward(self, features, flows=None, mode='fit'):
        """
        Args:
            features: B C H W
        """
        if mode == 'fit':
            point_coords, point_labels = self._sample_train_points_with_flow(features, flows)
            point_fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords
            )

            return point_logits, point_labels
        else:
            return F.sigmoid(self._subdivision_inference(features))

    def _sample_train_points_with_flow(self, features, flows):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        flows_mean = torch.mean(flows, dim=(2, 3), keepdim=True)
        flows_centered = flows - flows_mean
        flows_centered_norm = torch.norm(flows_centered, dim=1, keepdim=True)
        flows_centered_norm_max = flows_centered_norm.flatten(2).max(dim=2)[0].unsqueeze(1).unsqueeze(1)
        
        pos_masks = (flows_centered_norm > flows_centered_norm_max * 0.8).squeeze(1)
        neg_masks = (flows_centered_norm < flows_centered_norm_max * 0.2).squeeze(1)

        point_coords = []
        point_labels = []

        for(i, (pos_mask, neg_mask)) in enumerate(zip(pos_masks, neg_masks)):

            if torch.sum(neg_mask) == 0:
                neg_mask[0, 0] = True
            neg_idxs = torch.nonzero(neg_mask).flip(1)
            pos_idxs = torch.nonzero(pos_mask).flip(1)

            if len(pos_idxs) == 0:
                num_samples = self.mask_point_train_num_points
                while len(neg_idxs) < num_samples:
                    neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
                neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
                point_coords.append(neg_idxs)
                point_labels.append(torch.zeros(num_samples))

            else:
                num_samples = self.mask_point_train_num_points // 2
                while len(neg_idxs) < num_samples:
                    neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
                neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
                while len(pos_idxs) < num_samples:
                    pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
                pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples]
                point_coords.append(torch.cat([neg_idxs, pos_idxs], dim=0))
                point_labels.append(torch.cat([torch.zeros(num_samples), torch.ones(num_samples)], dim=0))

        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        return point_coords, point_labels

@MODELS.register_module()
class JBFHead(BaseHead):

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 num_layers: int = 3,
                 hid_channels: int = 256,
                 train_num_points: int = 128,
                 subdivision_steps: int = 3,
                 image_feature_enabled: bool = True,
                 pos_enc_enabled: bool = True,
                 scale: float = 1./4.,
                 use_flow: bool = False,
                 loss: ConfigType = dict(
                     type='MultipleLossWrapper',
                     losses=[
                         dict(type='BodySegTrainLoss', use_target_weight=True),
                         dict(type='JointSegTrainLoss', use_target_weight=True),
                         dict(type='BodySegTrainLoss', use_target_weight=True),
                     ]),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None
        self.flag = True
        self.flag2 = True
        self.use_flow = use_flow

        self.body_dec = nn.Sequential(
            Conv2d(in_channels, hid_channels, 3, 1, 1, activation=nn.ReLU()),
            nn.BatchNorm2d(hid_channels)
        )

        self.joint_dec = nn.Sequential(
            Conv2d(in_channels, hid_channels, 3, 1, 1, activation=nn.ReLU()),
            nn.BatchNorm2d(hid_channels)
        )

        if self.use_flow:
            self.flow_dec = nn.Sequential(
                Conv2d(in_channels, hid_channels, 3, 1, 1, activation=nn.ReLU()),
                nn.BatchNorm2d(hid_channels)
            )

        self.body_head = BodyMaskHead(hid_channels, train_num_points, subdivision_steps, scale, \
                                      num_layers, hid_channels, image_feature_enabled, pos_enc_enabled)
        self.joint_head = JointMaskHead(hid_channels, train_num_points, subdivision_steps, scale, \
                                        num_layers, hid_channels, image_feature_enabled, pos_enc_enabled, out_channels)
        if self.use_flow:
            self.flow_head = FlowMaskHead(hid_channels, train_num_points, subdivision_steps, scale, \
                                        num_layers, hid_channels, image_feature_enabled, pos_enc_enabled)
        
    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(
                type='Normal', layer='Linear', std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg
    
    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)
        
    def forward(self, feats: Tuple[Tensor], feats_flow=None, skls=None, gt_masks=None, flows=None, mode='fit') -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = feats[-1]
        x_body = self.body_dec(x)
        x_joint = self.joint_dec(x)
        if self.use_flow:
            x_flow = self.flow_dec(feats_flow[-1])

        skls_ = skls[:,0,...] if skls is not None else None
        gt_masks_body = gt_masks[:,0,...] if gt_masks is not None else None
        gt_masks_joints = gt_masks[:,1:,...] if gt_masks is not None else None

        ret_body = self.body_head(x_body, skls=skls_, gt_masks=gt_masks_body, mode=mode)
        ret_joint = self.joint_head(x_joint, skls=skls_, gt_masks=gt_masks_joints, mode=mode)
        if self.use_flow:
            ret_flow = self.flow_head(x_flow, flows=flows, mode=mode)

        if mode == 'fit':
            if self.use_flow:
                return ret_body, ret_joint, ret_flow
            return ret_body, ret_joint
        else:
            if self.use_flow:
                return torch.cat([ret_body, ret_joint, ret_flow], dim=1)
            return torch.cat([ret_body, ret_joint], dim=1)
    
    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {},
                feats_flow: Tuple[Tensor] = None) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        masks = self.forward(feats, feats_flow, mode='inference')

        preds = self.decode(masks)

        pred_fields = [
            PixelData(heatmaps=hm) for hm in masks.detach()
        ]

        return preds, pred_fields

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {},
             feats_flow: Tuple[Tensor] =None,
             flows: Tensor=None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """

        gt_masks = torch.stack(
            [d.gt_fields.masks for d in batch_data_samples])
        keypoints = None
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])  

        rets = self.forward(feats, feats_flow, keypoints, gt_masks, flows, mode='fit')

        input_list = []
        target_list = []

        for i, ret in enumerate(rets):
            input_list.append(ret[0])
            target_list.append(ret[1])

        # calculate losses
        losses = dict()

        loss_list = self.loss_module(input_list, target_list, keypoint_weights)

        loss = 0
        for loss_i in loss_list:
            loss += loss_i

        losses.update(loss_kpt=loss)

        return losses