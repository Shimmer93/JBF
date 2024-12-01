# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import PoseDataPreprocessor
from mmengine.utils import is_seq_of

from mmpose.registry import MODELS

@MODELS.register_module()
class PoseFlowDataPreprocessor(PoseDataPreprocessor):
    def __init__(self,
                 mean: Sequence[float] = None,
                 std: Sequence[float] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking,
            batch_augments=batch_augments)
        
    def forward(self, data: dict, training: bool = False) -> dict:
        _batch_inputs = data['inputs']
        if is_seq_of(_batch_inputs, torch.Tensor):
            _batch_inputs = torch.stack(_batch_inputs, dim=0)
        input1, input2 = _batch_inputs.chunk(2, dim=1)
        data['inputs'] = torch.cat([input1, input2], dim=0)
        data = super().forward(data=data, training=training)
        inputs = data['inputs']
        input1, input2 = inputs.chunk(2, dim=0)
        data['inputs'] = torch.cat([input1, input2], dim=1)
        return data