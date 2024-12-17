#!/bin/bash

#SBATCH --job-name=vis_jbf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu-share
#SBATCH --cpus-per-task=16

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/visualization.py \
    --mode image_grouped \
    --video-dir /home/zpengac/pose/PoseSegmentationMask/demo_videos_test \
    --jbf-dir inference/demo_image_grouped/jbfs \
    --anno-dir inference/demo_image_grouped/annos \
    --result-dir inference/demo_image_grouped/results \
    --label-map /home/zpengac/pose/pyskl/tools/data/label_map/nturgbd_120.txt \
    --out-dir visualization/demo_image_grouped
