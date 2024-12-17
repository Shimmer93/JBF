#!/bin/bash

#SBATCH --job-name=ifr_jbf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-share
#SBATCH --cpus-per-task=16
##SBATCH --nodelist=hhnode-ib-140

bash tools/dist_run.sh tools/inference.py 8 \
    --mode video_grouped \
    --det-config SNSNet/demo/mmdetection_cfg/faster_rcnn_person.py \
    --det-ckpt https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth \
    --skl-config SNSNet/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py \
    --skl-ckpt https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth \
    --jbf-config SNSNet/configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference_videobox-256x256.py \
    --jbf-ckpt /home/zpengac/pose/PoseSegmentationMask/logs/coco_cvpr/best_coco_AP_epoch_210.pth \
    --flow-ckpt /home/zpengac/pose/SNSNet/checkpoints/flow_ckpt.pth \
    --har-config JBFConv3D/configs/jbf/slowonly_r50_8xb16-u48-240e_ntu120-xsub-keypoint_inference_videobox.py \
    --har-ckpt /home/zpengac/pose/pyskl/work_dirs/psm/slowonly_r50_ntu120_xsub/joint_final6/best_top1_acc_epoch_24.pth \
    --video-dir /home/zpengac/pose/PoseSegmentationMask/demo_videos_test \
    --out-jbf-dir inference/demo_video_grouped/jbfs \
    --out-anno-dir inference/demo_video_grouped/annos \
    --out-result-dir inference/demo_video_grouped/results \
    --label-map /home/zpengac/pose/pyskl/tools/data/label_map/nturgbd_120.txt \
    --batched \
    --batch-size-det 16 \
    --batch-size-jbf 16