_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/scratch/PI/cqf/har_data/coco/'

# codec settings
codec = dict(
    type='JBFCodec', input_size=(256, 256), mask_size=(256, 256), dataset_type=dataset_type, sigma=3, use_flow=True)

# model settings
model = dict(
    type='SNSNet',
    data_preprocessor=dict(
        type='JBFDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    flownet=dict(
        type='RAFT',
        args_dict=dict(
            flow_model_path = '/home/zpengac/RAFT/models/raft-sintel.pth',
            global_flow = True,
            dataset = 'sintel',
            small = False
        )
    ),
    backbone_flow=dict(
        type='LiteHRNet',
        in_channels=2,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (32, 64),
                    (32, 64, 128),
                    (32, 64, 128, 256),
                )),
            with_head=True,
        )),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type='JBFHead',
        in_channels=32,
        out_channels=17,
        num_layers=3,
        hid_channels=64,
        train_num_points=256,
        subdivision_steps=2,
        scale=1/4,
        use_flow=True,
        loss=dict(type='MultipleLossWrapper', losses=[
             dict(type='BodySegTrainLoss', loss_weight=1, use_target_weight=True),
             dict(type='JointSegTrainLoss', loss_weight=0.5, neg_weight=0.95, use_target_weight=True)
             ]),
        decoder=codec),
    use_flow=True,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/home/zpengac/pose/PoseSegmentationMask/logs/jhmdb_final5/best_PCK_epoch_100.pth'))

find_unused_parameters = True

# pipelines
train_pipeline = [
    dict(type='LoadImagePair'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImagePair'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=data_root+'person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='JBFCocoMetric',
    outfile_prefix='logs/coco_final/td-hm_hrnet-w32_8xb64-210e_coco-256x192',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
