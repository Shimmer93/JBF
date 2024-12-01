# JBF
Official repo for paper "JBF: An Enhanced Representation of Skeleton for Video-based Human Action Recognition"

## SNSNet

### Installation
1. Create environment:
    ```
    conda create -n sns python=3.9 -y
    conda activate sns
    ```
2. Install PyTorch:
    ```
    conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
3. Install MMPose:
    ```
    pip install -U openmim
    mim install mmengine
    mim install "mmcv==2.1.0"
    mim install "mmdet>=3.1.0"
    mim install "mmpose>=1.1.0"
    ```