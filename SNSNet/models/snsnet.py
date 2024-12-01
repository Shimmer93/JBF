# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional, Tuple
from collections import OrderedDict

import torch
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from mmpose.models.pose_estimators.base import BasePoseEstimator

import matplotlib.pyplot as plt
import os

count = 0

@MODELS.register_module()
class SNSNet(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 flownet: OptConfigType = None,
                 backbone_flow: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 noflow_ckpt_path = None,
                 use_flow = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        # os.makedirs('./input', exist_ok=True)
        
        self.flownet = MODELS.build(flownet)
        self.backbone_flow = MODELS.build(backbone_flow)
        self.use_flow = use_flow                                                                            

        if noflow_ckpt_path is not None:
            ckpt = torch.load(noflow_ckpt_path, map_location='cpu')
            sd = ckpt['state_dict']
            new_sd = OrderedDict()
            for k, v in sd.items():
                if k.startswith('backbone'):
                    new_sd[k[9:]] = v
            self.backbone.load_state_dict(new_sd, strict=False)

    def extract_feat_without_flow(self, inputs: Tensor) -> Tensor:
        x_body = self.backbone(inputs)
        if self.with_neck:
            x_body = self.neck(x_body)
        return x_body

    def extract_feat_with_flow(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        global count
        count += 1
        # self.flownet.eval()

        x0, x1 = inputs[:, :3, ...], inputs[:, 3:, ...]

        with torch.no_grad():
            flow = self.flownet(x0, x1)[-1].detach()
            flow_mean = torch.mean(flow, dim=(2, 3), keepdim=True)
            flow_std = torch.std(flow, dim=(2, 3), keepdim=True)
            flow_ = (flow - flow_mean) / flow_std

        x_body = self.backbone(x0)
        if self.with_neck:
            x_body = self.neck(x_body)
        x_flow = self.backbone_flow(flow_.clone().detach())

        return x_body, x_flow, flow

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        losses = dict()

        if self.use_flow:
            feats_body, feats_flow, flows = self.extract_feat_with_flow(inputs)
        else:
            feats_body = self.extract_feat_without_flow(inputs)
            feats_flow = None
            flows = None

        if self.with_head:
            losses.update(
                self.head.loss(feats_body, data_samples, train_cfg=self.train_cfg, feats_flow=feats_flow, flows=flows))

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.use_flow:
            feats_body, feats_flow, _ = self.extract_feat_with_flow(inputs)
        else:
            feats_body = self.extract_feat_without_flow(inputs)
            feats_flow = None

        preds = self.head.predict(feats_body, data_samples, test_cfg=self.test_cfg, feats_flow=feats_flow)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            if 'input_center' in data_sample.metainfo:
                input_center = data_sample.metainfo['input_center']
                input_scale = data_sample.metainfo['input_scale']
                input_size = data_sample.metainfo['input_size']

                pred_instances.keypoints[..., :2] = \
                    pred_instances.keypoints[..., :2] / input_size * input_scale \
                    + input_center - 0.5 * input_scale
                
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples

@MODELS.register_module()
class SNSNetSmall(SNSNet):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 flownet: OptConfigType = None,
                 backbone_flow: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 noflow_ckpt_path = None,
                 use_flow = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            flownet=flownet,
            backbone_flow=backbone_flow,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo,
            noflow_ckpt_path=noflow_ckpt_path,
            use_flow=use_flow)
        
    def extract_feat_with_flow(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        global count
        count += 1
        # self.flownet.eval()
        b = inputs.shape[0]
        x0, x1 = inputs[:, :3, ...], inputs[:, 3:, ...]

        with torch.no_grad():
            if b % 4 != 0:
                x0_ = torch.nn.functional.interpolate(x0, scale_factor=0.5, mode='bilinear', align_corners=False)
                x1_ = torch.nn.functional.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)
                flow = self.flownet(x0_, x1_)[-1].detach()
            else:
                x0_ = torch.nn.functional.interpolate(x0, scale_factor=0.25, mode='bilinear', align_corners=False)
                x1_ = torch.nn.functional.interpolate(x1, scale_factor=0.25, mode='bilinear', align_corners=False)
                _, c, h, w = x0_.shape

                x0_ = x0_.reshape(b//4, 2, 2, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b//4, c, 2*h, 2*w)
                x1_ = x1_.reshape(b//4, 2, 2, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b//4, c, 2*h, 2*w)
                    
                flow = self.flownet(x0_, x1_)[-1].detach()
                flow = flow.reshape(b//4, 2, 2, h, 2, w).permute(0, 2, 4, 1, 3, 5).reshape(b, 2, h, w)
            flow_mean = torch.mean(flow, dim=(2, 3), keepdim=True)
            flow_std = torch.std(flow, dim=(2, 3), keepdim=True)
            flow_ = (flow - flow_mean) / flow_std

        if b % 4 != 0:
            flow_ = torch.nn.functional.interpolate(flow_, scale_factor=2, mode='bilinear', align_corners=False)
            flow = torch.nn.functional.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            flow_ = torch.nn.functional.interpolate(flow_, scale_factor=4, mode='bilinear', align_corners=False)
            flow = torch.nn.functional.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=False)
        flow_ = torch.cat((flow_, x0), dim=1)

        x_body = self.backbone(x0)
        if self.with_neck:
            x_body = self.neck(x_body)
        x_flow = self.backbone_flow(flow_.clone().detach())

        return x_body, x_flow, flow