import os
import os.path as osp
import numpy as np
import cv2
import argparse
from glob import glob
from tqdm import tqdm

from mmengine import load, dump
from mmaction.models.utils import Graph
from mmaction.datasets.transforms.jbf_tranforms import JBFCompactResizePad, JBFDecode
# import decord

def fall_detection_action_text_mapper(action):
    if action.startswith('standing'):
        action = 'standing up'
    elif action.startswith('sitting'):
        action = 'sitting down'
    elif action.startswith('squat'):
        action = 'squatting down'
    elif action.startswith('walking'):
        action = 'walking'
    elif action.startswith('falling'):
        action = 'falling'
    elif action.startswith('staggering'):
        action = 'staggering'
    elif action.startswith('jump'):
        action = 'jumping'
    else:
        action = 'other action'
    return action

def fall_detection_action_color_mapper(action):
    if action == 'falling':
        return (0, 0, 255)
    return (0, 0, 0)

def update_grouped(canvas, frame, jbf, skl, skl_score, result, label_map, bones, colors, 
                   action_text_mapper=None, action_color_mapper=None, raw_jbv = False, s = 256, m = 32, p = 8):
    action = label_map[result[0]]
    if action_text_mapper is not None:
        action = action_text_mapper(action)
    action_color = (0, 0, 0)
    if action_color_mapper is not None:
        action_color = action_color_mapper(action)
    if len(action) > 28:
        action = action[:28]+'...'
    conf = result[1] * 100

    canvas[0:m+2, :] = 255
    cv2.putText(canvas, f'JBF Pred: {action}', (p, m-6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, action_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f'Conf: {conf:.4f}%', (s*2+p*3, m-6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    (height, width) = frame.shape[:2]
    ratio = float(s) / height
    dsize = (int(width * ratio), s)
    frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)
    canvas[m*2:s+m*2, p:frame.shape[1]+p] = frame

    joint_map = np.zeros((s, s, 3), dtype=np.uint8)
    skl_map = np.zeros((s, s, 3), dtype=np.uint8)

    body_map = jbf[..., 17]
    flow_map = jbf[..., 18]
    body_map = cv2.resize(body_map, (s, s), interpolation=cv2.INTER_NEAREST)
    flow_map = cv2.resize(flow_map, (s, s), interpolation=cv2.INTER_NEAREST)

    if raw_jbv:
        joint_map_small = np.zeros((s//4, s//4, 3), dtype=np.uint8)
        for i in range(17):
            joint_map[jbf[..., i] > 0] = colors[i]*0.5
        joint_map = cv2.resize(joint_map_small, (s, s), interpolation=cv2.INTER_NEAREST)

    for skl_i, skl_score_i in zip(skl, skl_score):
        if skl_score_i.max() < 0.5:
            continue
        for bone in bones:
            if skl_score_i[bone[0]] < 0.5 or skl_score_i[bone[1]] < 0.5:
                continue
            start = tuple((skl_i[bone[0]]).astype(int))
            end = tuple((skl_i[bone[1]]).astype(int))
            color = (int((colors[bone[0]][0]+colors[bone[1]][0])//2), int((colors[bone[0]][1]+colors[bone[1]][1])//2), int((colors[bone[0]][2]+colors[bone[1]][2])//2))
            cv2.line(joint_map, start, end, color, 3)
            cv2.line(skl_map, start, end, color, 3)

    if not raw_jbv:
        for skl_i, skl_score_i in zip(skl, skl_score):
            if skl_score_i.max() < 0.5:
                continue
            for i in range(17):
                if skl_score_i[i] < 0.5:
                    continue
                center = tuple((skl_i[i]).astype(int))
                radius = int(np.sqrt(np.sum((jbf[..., i]>0).astype(np.int32))))
                color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
                cv2.circle(joint_map, center, radius, color, -1)

    canvas[m*2:s+m*2, s*2+p*3:s*3+p*3] = joint_map
    canvas[m*2:s+m*2, s*3+p*4:s*4+p*4] = np.expand_dims(body_map, axis=-1)
    canvas[m*2:s+m*2, s*4+p*5:s*5+p*5] = np.expand_dims(flow_map, axis=-1)

def create_inference_video_grouped(filename, video_file, jbf_dir, anno_dir, result_dir, label_map_file, save_dir,
                                   bones, colors, transforms=None, action_text_mapper=None, action_color_mapper=None, 
                                   raw_jbv = False, s = 256, m = 32, p = 8):
    print('Position 1')
    video_reader = cv2.VideoCapture(video_file)
    frame_rate = video_reader.get(cv2.CAP_PROP_FPS)

    print('Position 2')

    anno = load(osp.join(anno_dir, filename+'.pkl'))
    anno['jbf_path'] = osp.join(jbf_dir, filename+'.npy')
    anno = JBFDecode()(anno)
    if transforms is not None:
        anno = transforms(anno)

    print('Position 3')

    skls = anno['keypoint']
    skl_scores = anno['keypoint_score']
    jbfs = anno['imgs']
    num_frames = len(jbfs)

    result = load(osp.join(result_dir, filename+'.pkl'))
    label_map = [x.strip() for x in open(label_map_file).readlines()]

    canvas = np.ones((s+m*2+p, s*5+p*6, 3), dtype=np.uint8) * 255
    os.makedirs(save_dir, exist_ok=True)
    result_video = cv2.VideoWriter(osp.join(save_dir, filename+'.mp4'), 
                                   cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (s*5+p*6, s+m*2+p))
    
    print('Position 4')

    cv2.putText(canvas, 'Video', (p, m*2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Joint Map Volume', (s*2+p*3, m*2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Body Map', (s*3+p*4, m*2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Flow Map', (s*4+p*5, m*2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    
    for i in range(num_frames):
        frame = video_reader.read()[1]
        print('Position 5')
        update_grouped(canvas, frame, jbfs[i], skls[:, i], skl_scores[:, i], result[i], label_map, bones, colors,
                                action_text_mapper, action_color_mapper, raw_jbv, s, m, p)
        result_video.write(canvas)
        print('Position 6')

    result_video.release()

    print('Position 7')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument('--mode', type=str, choices=['video_grouped', 'image_grouped', 'image_individual'], default='all_in_one',
                        help='mode for generating JBF. video_grouped: use the box of the whole video and all people in the video'
                             'image_grouped: use the box of each image and all people in the image'
                             'image_individual: use the box of each image and each person in the image')
    parser.add_argument('--video-dir', type=str, help='input video directory')
    parser.add_argument('--jbf-dir', type=str, help='JBF directory')
    parser.add_argument('--anno-dir', type=str, help='annotation directory')
    parser.add_argument('--result-dir', type=str, help='result directory')
    parser.add_argument('--out-dir', type=str, help='output video directory')
    parser.add_argument('--label-map', type=str, help='label map file')
    parser.add_argument('--rescale-ratio', type=float, help='rescale ratio for JBF', default=4.0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('Loading videos...')
    video_fns = glob(osp.join(args.video_dir, '*.mp4')) + \
                glob(osp.join(args.video_dir, '*.avi')) + \
                glob(osp.join(args.video_dir, '*.mkv')) + \
                glob(osp.join(args.video_dir, '*.mov'))
    video_fns = sorted(video_fns)

    colors = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 128, 0], [128, 0, 255], [128, 64, 0], [255, 128, 255],
        [128, 128, 128], [0, 128, 0], [0, 0, 128], [128, 128, 0], [0, 128, 128], [128, 0, 128], [128, 64, 0], [64, 0, 128], [64, 32, 0], [128, 64, 128]
    ])

    print('Before Graph')

    graph = Graph('coco')
    bones = graph.inward

    print('After Graph')

    for video_fn in tqdm(video_fns):
        filename = video_fn.split('/')[-1].split('.')[0]
        if args.mode == 'video_grouped': 
            print('Before create_inference_video_grouped')
            create_inference_video_grouped(filename, video_fn, args.jbf_dir, args.anno_dir, args.result_dir, args.label_map, 
                                        args.out_dir, bones, colors, None, fall_detection_action_text_mapper, fall_detection_action_color_mapper, raw_jbv=False)
        elif args.mode == 'image_grouped':
            print('Before create_inference_video_grouped')
            compact_resize_pad = JBFCompactResizePad()
            create_inference_video_grouped(filename, video_fn, args.jbf_dir, args.anno_dir, args.result_dir, args.label_map, 
                                        args.out_dir, bones, colors, compact_resize_pad, fall_detection_action_text_mapper, fall_detection_action_color_mapper, raw_jbv=True)
        else:
            print('Not implemented yet')
            break

if __name__ == '__main__':
    main()