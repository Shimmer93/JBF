import argparse
import copy as cp
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from tqdm import tqdm
from glob import glob

from mmengine import load, dump
from mmengine.dist.utils import get_dist_info, init_dist
from mmaction.apis import init_recognizer
from mmpose.utils import save_jbf_seq, JBFInferenceCompact, JBFInferenceResize

from SNSNet.tools.inference_jbf import load_models as load_sns_models, extract_frame
from SNSNet.tools.inference_jbf import det_inference, skl_inference, jbf_inference
from JBFConv3D.tools.inference_jbf import inference_per_frame as har_inference_per_frame, inference_per_video as har_inference_per_video

default_det_config = 'SNSNet/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
default_det_ckpt = 'SNSNet/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
default_jbf_config = 'SNSNet/configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py'
default_jbf_ckpt = 'SNSNet/checkpoints/snsnet_img.pth'
default_flow_ckpt = 'SNSNet/checkpoints/snsnet_flow.pth'
default_skl_config = 'SNSNet/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
default_skl_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'
default_har_config = 'JBFConv3D/configs/jbf/slowonly_r50_8xb16-u48-240e_ntu120-xsub-keypoint_inference.py'
default_har_ckpt = 'JBFConv3D/checkpoints/jbfconv3d_ntu120.pth'


def load_models(det_config, det_ckpt, skl_config, skl_ckpt, 
                jbf_config, jbf_ckpt, flow_ckpt, har_config, har_ckpt):
    det_model, skl_model, jbf_model = load_sns_models(det_config, det_ckpt, skl_config, skl_ckpt, 
                                                      jbf_config, jbf_ckpt, flow_ckpt)
    har_model = init_recognizer(har_config, har_ckpt, 'cuda')

    return det_model, skl_model, jbf_model, har_model

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--skl-config', type=str, default=default_skl_config)
    parser.add_argument('--skl-ckpt', type=str, default=default_skl_ckpt)
    parser.add_argument('--jbf-config', type=str, default=default_jbf_config)
    parser.add_argument('--jbf-ckpt', type=str, default=default_jbf_ckpt)
    parser.add_argument('--flow-ckpt', type=str, default=default_flow_ckpt)
    parser.add_argument('--har-config', type=str, default=default_har_config)
    parser.add_argument('--har-ckpt', type=str, default=default_har_ckpt)
    parser.add_argument('--video-dir', type=str, help='input video directory')
    parser.add_argument('--out-jbf-dir', type=str, help='output JBF directory')
    parser.add_argument('--out-anno-dir', type=str, help='output annotation directory')
    parser.add_argument('--out-result-dir', type=str, help='output result directory')
    parser.add_argument('--label-map', type=str, help='label map file', default='tools/data/label_map/nturgbd_120.txt')
    parser.add_argument('--rescale-ratio', type=float, help='rescale ratio for JBF', default=4.0)
    parser.add_argument('--batched', action='store_true', help='whether to use batched inference')
    parser.add_argument('--per-video', action='store_true', help='predict action per video instead of per frame')
    parser.add_argument('--batch-size-outer', type=int, default=64)
    parser.add_argument('--batch-size-det', type=int, default=16)
    parser.add_argument('--batch-size-jbf', type=int, default=16)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local-rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    
    print('Loading videos...')
    video_fns = glob(osp.join(args.video_dir, '*.mp4')) + \
                glob(osp.join(args.video_dir, '*.avi')) + \
                glob(osp.join(args.video_dir, '*.mkv')) + \
                glob(osp.join(args.video_dir, '*.mov'))
    video_fns = sorted(video_fns)
    annos = [{'filename': fn, 'frame_dir': fn.split('/')[-1].split('.')[0]} for fn in video_fns]

    print('Initializing distributed environment...')
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

    print('Loading models...')
    det_model, skl_model, jbf_model, har_model = load_models(args.det_config, args.det_ckpt, args.skl_config, args.skl_ckpt, 
                                                      args.jbf_config, args.jbf_ckpt, args.flow_ckpt, args.har_config, args.har_ckpt)

    compact = JBFInferenceCompact(padding=0.25, threshold=10, hw_ratio=(1., 1.), allow_imgpad=True)
    resize = JBFInferenceResize(scale=(256, 256), keep_ratio=False, interpolation='bilinear')

    os.makedirs(args.out_jbf_dir, exist_ok=True)
    os.makedirs(args.out_anno_dir, exist_ok=True)
    os.makedirs(args.out_result_dir, exist_ok=True)

    print('Inference...')
    for anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        frame_dir = anno['frame_dir']

        batch_frames = [frames[i:i+args.batch_size_outer] for i in range(0, len(frames), args.batch_size_outer)]

        all_jbf_seqs = []
        all_new_annos = []
        all_det_results = []

        for i, batch in enumerate(batch_frames):
            det_results = det_inference(det_model, batch, batch_size_det=args.batch_size_det)
            # * Get detection results for human
            for j, det_sample in enumerate(det_results):
                # * filter boxes with small scores
                res = det_sample.pred_instances.bboxes.cpu().numpy()
                scores = det_sample.pred_instances.scores.cpu().numpy()
                res = res[scores >= args.det_score_thr]
                # * filter boxes with small areas
                box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
                assert np.all(box_areas >= 0)
                res = res[box_areas >= args.det_area_thr]
                det_results[j] = res
            all_det_results.extend(det_results)

        total_frames = len(frames)
        num_person = max([len(x) for x in all_det_results])
        anno['total_frames'] = total_frames
        anno['num_person_raw'] = num_person

        batch_det_results = [all_det_results[j:j+args.batch_size_outer] for j in range(0, len(all_det_results), args.batch_size_outer)]

        for i, (batch, det_results) in enumerate(zip(batch_frames, batch_det_results)):
            new_anno = cp.deepcopy(anno)

            shape = batch[0].shape[:2]
            new_anno['img_shape'] = shape
            new_anno = skl_inference(new_anno, skl_model, batch, det_results)

            all_new_annos.append(new_anno)
        
        anno['keypoint'] = np.concatenate([x['keypoint'] for x in all_new_annos], axis=1)
        anno['keypoint_score'] = np.concatenate([x['keypoint_score'] for x in all_new_annos], axis=1)
        anno['img_shape'] = all_new_annos[0]['img_shape']
        anno['modality'] = 'Pose'
        anno['label'] = -1
        anno['imgs'] = frames
        anno.pop('filename')

        anno = compact(anno)
        anno = resize(anno)

        frames = anno['imgs']
        batch_frames = [frames[j:j+args.batch_size_outer] for j in range(0, len(frames), args.batch_size_outer)]

        for i, (batch, det_results) in enumerate(zip(batch_frames, batch_det_results)):
            batch_next = cp.deepcopy(batch)
            batch_next.pop(0)
            batch_next.append(batch[-1] if i + 1 == len(batch_frames) else batch_frames[i+1][0])
            batch = np.concatenate([batch, batch_next], axis=-1)

            jbf_seq = jbf_inference(jbf_model, batch, det_results, args.rescale_ratio, args.batched, args.batch_size_jbf)
            all_jbf_seqs.extend(jbf_seq)

        anno.pop('imgs')
        out_anno_fn = osp.join(args.out_anno_dir, f'{frame_dir}.pkl')
        dump(anno, out_anno_fn)
        out_fn = osp.join(args.out_jbf_dir, f'{frame_dir}.npy')
        save_jbf_seq(jbf_seq, out_fn)

        anno['start_index'] = 0
        anno['clip_len'] = 48
        anno['frame_interval'] = None
        anno['num_clips'] = 1
        anno['jbf_path'] = out_fn

        if args.per_video:
            max_pred_index, max_pred_score = har_inference_per_video(har_model, anno)
            results = [(max_pred_index, max_pred_score)]
        else:
            results = har_inference_per_frame(har_model, anno)

        out_result_fn = osp.join(args.out_result_dir, f'{frame_dir}.pkl')
        dump(results, out_result_fn)

if __name__ == '__main__':
    main()