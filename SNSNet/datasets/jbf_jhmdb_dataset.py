# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from copy import deepcopy
from itertools import chain, filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.logging import MessageHub
from mmengine.utils import is_list_of
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.datasets.base import BaseCocoStyleDataset

import os.path as osp
from typing import Optional

import numpy as np

@DATASETS.register_module()
class JBFJhmdbDataset(BaseCocoStyleDataset):
    """JhmdbDataset dataset for pose estimation.

    "Towards understanding action recognition", ICCV'2013.
    More details can be found in the `paper
    <https://openaccess.thecvf.com/content_iccv_2013/papers/\
    Jhuang_Towards_Understanding_Action_2013_ICCV_paper.pdf>`__

    sub-JHMDB keypoints::

        0: "neck",
        1: "belly",
        2: "head",
        3: "right_shoulder",
        4: "left_shoulder",
        5: "right_hip",
        6: "left_hip",
        7: "right_elbow",
        8: "left_elbow",
        9: "right_knee",
        10: "left_knee",
        11: "right_wrist",
        12: "left_wrist",
        13: "right_ankle",
        14: "left_ankle"

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/jhmdb.py')

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""

        assert exists(self.ann_file), (
            f'Annotation file `{self.ann_file}`does not exist')

        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        if 'categories' in self.coco.dataset:
            self._metainfo['CLASSES'] = self.coco.loadCats(
                self.coco.getCatIds())

        instance_list = []
        image_list = []

        for img_id in self.coco.getImgIds():
            if img_id % self.sample_interval != 0:
                continue
            img = self.coco.loadImgs(img_id)[0]

            img_path = osp.join(self.data_prefix['img'], img['file_name'])
            frm_id = int(osp.basename(img_path).split('.')[0])
            nframes = img['nframes']
            frm_id2 = frm_id + 1 if frm_id < nframes else frm_id
            img_id2 = img_id + 1 if frm_id < nframes else img_id
            img_path2 = img_path.replace(f'{frm_id:05d}', f'{frm_id2:05d}')

            img.update({
                'img_id': img_id,
                'img_path': img_path,
                'img_id2': img_id2,
                'img_path2': img_path2,
            })
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            ann_ids2 = self.coco.getAnnIds(imgIds=img_id2)
            for ann, ann2 in zip(self.coco.loadAnns(ann_ids), self.coco.loadAnns(ann_ids2)):

                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_ann2_info=ann2, raw_img_info=img))

                # skip invalid instance annotation.
                if not instance_info:
                    continue

                instance_list.append(instance_info)
        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        ann2 = raw_data_info['raw_ann2_info']
        img = raw_data_info['raw_img_info']

        img_path = osp.join(self.data_prefix['img'], img['file_name'])
        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        # JHMDB uses matlab format, index is 1-based,
        # we should first convert to 0-based index
        x -= 1
        y -= 1
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        # JHMDB uses matlab format, index is 1-based,
        # we should first convert to 0-based index
        keypoints = _keypoints[..., :2] - 1
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if not self.test_mode:
            _keypoints2 = np.array(
                ann2['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            keypoints2 = _keypoints2[..., :2] - 1
            keypoints_visible2 = np.minimum(1, _keypoints2[..., 2])

            keypoints = np.concatenate([keypoints, keypoints2], axis=-2)
            keypoints_visible = np.concatenate(
                [keypoints_visible, keypoints_visible2], axis=-1)
        
        num_keypoints = np.count_nonzero(keypoints.max(axis=2))
        area = np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None)
        category_id = ann.get('category_id', [1] * len(keypoints))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img_path,
            'img_id2': img['img_id2'],
            'img_path2': img['img_path2'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'area': np.array(area, dtype=np.float32),
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': category_id,
            'flag': 0,
        }

        return data_info
