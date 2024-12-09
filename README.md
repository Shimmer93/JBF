# JBF
Official repo for paper "JBF: An Enhanced Representation of Skeleton for Video-based Human Action Recognition"

## Installation
1. Create environment:
    ```
    conda create -n jbf python=3.9 -y
    conda activate jbf
    ```
2. Install PyTorch (Change verions according to your CUDA version):
    ```
    conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
3. Install OpenMM:
    ```
    pip install -U openmim
    mim install mmengine "mmcv==2.1.0" "mmdet==3.2.0"
    ```
4. Install SNSNet:
    ```
    cd SNSNet
    pip install -r requirements.txt
    pip install -v -e .
    cd ..
    ```
4. Install JBFConv3D:
    ```
    cd JBFConv3D
    pip install -v -e .
    cd ..
    ```
5. (Optional, bug exists with certain CUDA versions) Install alternate correlation implementation from [RAFT]('https://github.com/princeton-vl/RAFT/tree/master'):
    ```
    cd SNSNet/mmpose/models/flownets/alt_cuda_corr
    python setup.py install
    cd ../../../../..
    ```