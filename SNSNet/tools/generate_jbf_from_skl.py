# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.distributed as dist
from mmengine.dist.utils import get_dist_info, init_dist
from tqdm import tqdm
import cv2
from PIL import Image
import decord
from time import time
import pickle
from collections import OrderedDict
from mmpose.structures.bbox import get_warp_matrix
import io

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))

try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    # import mmpose  # noqa: F401
    from mmpose.apis import inference_topdown_batch, inference_topdown, init_model
    from mmpose.structures import PoseDataSample
    from mmpose.utils import adapt_mmdet_pipeline
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

# anno_path = '/scratch/PI/cqf/har_data/pkls/ntu60_rtmpose.pkl'
# default_det_config = 'demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py'
# default_det_ckpt = (
#     'logs/coco_final/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
# default_pose_config = 'configs/body_2d_keypoint/topdown_psm_flow/coco/td-hm_litehrnet-w32_8xb64-210e_coco-256x192_smurf_inference.py'
# default_pose_ckpt = (
#     'logs/coco_cvprlite/best_coco_AP_epoch_210.pth')
# default_flow_ckpt = (
#     'logs/jhmdb_cvprlite4/best_PCK_epoch_60.pth')
anno_path = '/scratch/PI/cqf/har_data/pkls/ntu120_hrnet.pkl'
default_det_config = 'demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py'
default_det_ckpt = (
    'logs/coco_final/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
default_pose_config = 'configs/body_2d_keypoint/topdown_psm_flow/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_inference.py'
default_pose_ckpt = (
    'logs/coco_cvpr/best_coco_AP_epoch_210.pth')
default_flow_ckpt = (
    'logs/jhmdb_new8/best_PCK_epoch_73.pth')

def get_bboxes_from_skeletons(skls, H, W, padding=0.25, threshold=10, hw_ratio=(1.,1.), allow_imgpad=True):
    # skls: N T J 2
    N, T, _, _ = skls.shape
    skls[np.isnan(skls)] = 0.
    kp_x = skls[..., 0]
    kp_y = skls[..., 1]

    min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
    min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
    max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
    max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

    # The compact area is too small
    if max_x - min_x < threshold or max_y - min_y < threshold or \
        max_x == -np.Inf or max_y == -np.Inf or min_x == np.Inf or min_y == np.Inf:
        min_x = 0
        min_y = 0
        max_x = W
        min_y = H
    else:
        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + padding)
        half_height = (max_y - min_y) / 2 * (1 + padding)

        if hw_ratio is not None:
            half_height = max(hw_ratio[0] * half_width, half_height)
            half_width = max(1 / hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        if not allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(W, max_x)), int(min(H, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

    bbox = np.array([[[min_x, min_y, max_x, max_y]]])
    bbox = np.repeat(bbox, N, axis=0)
    bboxes = np.repeat(bbox, T, axis=1)

    bboxes = bboxes.transpose(1, 0, 2)
    bboxes = [x for x in bboxes]

    return bboxes

def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]

def write_psm(save_path, joint_masks, body_mask=None, obj_mask=None, rescale_ratio=1.0):


    out_masks = joint_masks
    if body_mask is not None:
        out_masks = np.concatenate([out_masks, np.expand_dims(body_mask, axis=0)], axis=0)
    if obj_mask is not None:
        out_masks = np.concatenate([out_masks, np.expand_dims(obj_mask, axis=0)], axis=0)

    h, w = out_masks.shape[-2:]
    out_masks = np.stack([cv2.resize(mask, dsize=(int(w/rescale_ratio), int(h/rescale_ratio)), interpolation=cv2.INTER_LINEAR) for mask in out_masks])
    out_masks = (out_masks > 0).astype(np.uint8)
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
    
    psm = Image.fromarray(canvas)
    return psm

def write_psm_from_pose_sample(save_path, pose_sample: PoseDataSample, rescale_ratio=1.0):
    masks = pose_sample.pred_fields.heatmaps.detach().cpu()

    mask_body = (masks[0] > 0.5).float()
    mask_body = mask_body.numpy()
    mask_joints = (masks[1:-1] > 0.5).float()
    mask_flow = (masks[-1] > 0.5).float()
    mask_flow = mask_flow.numpy()
    mask_joints = mask_joints.numpy()

    psm = write_psm(save_path, mask_joints, mask_body, mask_flow, rescale_ratio=rescale_ratio)
    return psm

def pose_inference(anno_in, model, frames, det_results, compress=False, batch_size_pose=16):
    anno = cp.deepcopy(anno_in)
    fn = osp.join('/scratch/PI/cqf/har_data/ntu/psm5', anno['frame_dir'] + '.npy')
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = num_person

    data = list(zip(frames, det_results))
    # batches = [data[i:i+batch_size_pose] for i in range(0, len(frames), batch_size_pose)]
    batches = [data[i:i+batch_size_pose] for i in range(0, len(frames), batch_size_pose)]
    pose_samples = []
    for batch in batches:
        batch_frames, batch_det_results = zip(*batch)

        batch_pose_samples = inference_topdown_batch(model, batch_frames, batch_det_results, bbox_format='xyxy')
        pose_samples.extend(batch_pose_samples)

    masks = []
    for i, pose_sample in enumerate(pose_samples):
        save_path = anno['filename'].replace('_rgb.avi', f'/{i:03d}.png').replace('videos', 'psm5')
        mask_img = write_psm_from_pose_sample(save_path, pose_sample, rescale_ratio=4.0)

        mask = io.BytesIO()
        mask_img.save(mask, format='PNG')
        mask = np.frombuffer(mask.getvalue(), dtype=np.uint8)
        masks.append(mask)

    np.save(fn, np.array(masks, dtype=object), allow_pickle=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    # parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    # parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local-rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    parser.add_argument('--compress', action='store_true', help='whether to do K400-style compression')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs('/scratch/PI/cqf/har_data/ntu/psm5', exist_ok=True)
    with open(anno_path, 'rb') as f:
        annos = pickle.load(f)['annotations']
    for anno in annos:
        anno['filename'] = f'/scratch/PI/cqf/har_data/ntu/nturgb+d_rgb/{anno["frame_dir"]}_rgb.avi'
        anno['bboxes'] = get_bboxes_from_skeletons(anno['keypoint'], anno['img_shape'][0], anno['img_shape'][1])

    print('Loading models...')
    if args.non_dist:
        my_part = annos
        os.makedirs(args.tmpdir, exist_ok=True)
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        if rank == 0:
            os.makedirs(args.tmpdir, exist_ok=True)
        dist.barrier()
        my_part = annos[rank::world_size]

    pose_model = init_model(args.pose_config, args.pose_ckpt, 'cuda')
    flow_sd = torch.load(default_flow_ckpt, map_location='cpu')['state_dict']
    sd_backbone_flow = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('backbone_flow'):
            sd_backbone_flow[k.replace('backbone_flow.', '')] = v
    sd_flow_dec = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('head.flow_dec'):
            sd_flow_dec[k.replace('head.flow_dec.', '')] = v
    sd_flow_head = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('head.flow_head'):
            sd_flow_head[k.replace('head.flow_head.', '')] = v
    print('Loading flow model...')
    pose_model.backbone_flow.load_state_dict(sd_backbone_flow)
    pose_model.head.flow_dec.load_state_dict(sd_flow_dec)
    pose_model.head.flow_head.load_state_dict(sd_flow_head)

    print('Start inference...')
    results = []
    for anno in tqdm(my_part):
        if not osp.exists(anno['filename']):
            continue
        frames = extract_frame(anno['filename'])
        frames_next = cp.deepcopy(frames)
        frames_next.pop(0)
        frames_next.append(frames[-1])
        frames = [np.concatenate([frames[i], frames_next[i]], axis=-1) for i in range(len(frames))] 

        det_results = anno['bboxes']

        n_frames = min(len(frames), len(det_results))
        frames = frames[:n_frames]
        det_results = det_results[:n_frames]

        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        pose_inference(anno, pose_model, frames, det_results, compress=args.compress, batch_size_pose=16)

if __name__ == '__main__':
    main()