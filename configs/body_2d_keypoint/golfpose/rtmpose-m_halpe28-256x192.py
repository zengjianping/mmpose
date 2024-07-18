_base_ = ['../../_base_/default_runtime.py']

# common setting
num_keypoints = 28
input_size = (192, 256)

# runtime
max_epochs = 100
stage2_num_epochs = 20
base_lr = 4e-3
train_batch_size = 128
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            #checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',  # noqa
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple([s // 32 for s in input_size]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'Halpe28Dataset'
data_mode = 'topdown'
data_root = 'data/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# mapping
coco_halpe28 = [(i, i) for i in range(17)] + [(17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)] + [(100, 26), (121, 27)]

halpe28_halpe28 = [(i, i) for i in range(28)]

halpe_halpe28 = [(i, i) for i in range(26)] + [(103, 26), (124, 27)]

aic_halpe28 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7),
               (5, 9), (6, 12), (7, 14), (8, 16), (9, 11), (10, 13), (11, 15),
               (12, 17), (13, 18)]

crowdpose_halpe28 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16), (12, 17),
                     (13, 18)]

mpii_halpe28 = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (8, 18),
    (9, 17),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

jhmdb_halpe28 = [
    (0, 18),
    (2, 17),
    (3, 6),
    (4, 5),
    (5, 12),
    (6, 11),
    (7, 8),
    (8, 7),
    (9, 14),
    (10, 13),
    (11, 10),
    (12, 9),
    (13, 16),
    (14, 15),
]

ochuman_halpe28 = [(i, i) for i in range(17)]

posetrack_halpe28 = [
    (0, 0),
    (2, 17),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
]

# train datasets
dataset_golfdb = dict(
    type='Halpe28Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='golfdb/annotations/halpe28/train.json',
    data_prefix=dict(img='golfdb/images'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe28_halpe28)
    ],
)

dataset_ezgolf = dict(
    type='Halpe28Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240418/annotations/halpe28/train.json',
    data_prefix=dict(img='ezgolf/task_20240418/images'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe28_halpe28)
    ],
)

dataset_ezgolf_t2 = dict(
    type='Halpe28Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240628/annotations/halpe28/train.json',
    data_prefix=dict(img='ezgolf/task_20240628/images'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe28_halpe28)
    ],
)

dataset_coco = dict(
    type='CocoWholeBodyDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=coco_halpe28)
    ],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_v1.json',
    data_prefix=dict(img='halpe/hico_20160224_det/images/train2015'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe_halpe28)
    ],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=aic_halpe28)
    ],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='crowdpose/images/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=crowdpose_halpe28)
    ],
)

dataset_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='mpii/images/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=mpii_halpe28)
    ],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_train.json',
    data_prefix=dict(img='jhmdb/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=jhmdb_halpe28)
    ],
)

dataset_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_train.json',
    data_prefix=dict(img='posetrack18/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=posetrack_halpe28)
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    #num_batch_per_epoch=10,
    pin_memory=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/halpe28.py'),
        datasets=[
            dataset_golfdb,
            dataset_ezgolf,
            dataset_ezgolf_t2,
            dataset_coco,
            dataset_halpe,
            dataset_aic,
            #dataset_crowdpose,
            dataset_mpii,
            dataset_jhmdb,
            #dataset_posetrack,
        ],
        pipeline=train_pipeline,
        test_mode=False,
    )
)

# val datasets
val_golfdb = dict(
    type='Halpe28Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='golfdb/annotations/halpe28/val.json',
    data_prefix=dict(img='golfdb/images'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe28_halpe28)
    ],
)

val_ezgolf = dict(
    type='Halpe28Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240418/annotations/halpe28/val.json',
    data_prefix=dict(img='ezgolf/task_20240418/images'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe28_halpe28)
    ],
)

val_ezgolf_t2 = dict(
    type='Halpe28Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240628/annotations/halpe28/val.json',
    data_prefix=dict(img='ezgolf/task_20240628/images'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe28_halpe28)
    ],
)

val_coco = dict(
    type='CocoWholeBodyDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_val_v1.0.json',
    data_prefix=dict(img='coco/val2017/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=coco_halpe28)
    ],
)

val_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_val_v1.json',
    data_prefix=dict(img='coco/val2017/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe_halpe28)
    ],
)

val_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_val.json',
    data_prefix=dict(
        img='aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=aic_halpe28)
    ],
)

val_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_test.json',
    data_prefix=dict(img='crowdpose/images/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=crowdpose_halpe28)
    ],
)

val_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_val.json',
    data_prefix=dict(img='mpii/images/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=mpii_halpe28)
    ],
)

val_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_test.json',
    data_prefix=dict(img='jhmdb/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=jhmdb_halpe28)
    ],
)

val_ochuman = dict(
    type='OCHumanDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ochuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json',
    data_prefix=dict(img='ochuman/images/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=ochuman_halpe28)
    ],
)

val_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_val.json',
    data_prefix=dict(img='posetrack18/'),
    pipeline=[
        dict(type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=posetrack_halpe28)
    ],
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    #num_batch_per_epoch=10,
    pin_memory=False,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/halpe28.py'),
        datasets=[
            val_golfdb,
            val_ezgolf,
            val_ezgolf_t2,
            val_coco,
            val_halpe,
            val_aic,
            #val_crowdpose,
            val_mpii,
            val_jhmdb,
            #val_ochuman,
            #val_posetrack,
        ],
        pipeline=val_pipeline,
        test_mode=True,
    )
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='AUC', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = [dict(type='PCKAccuracy', thr=0.1), dict(type='AUC')]
test_evaluator = val_evaluator
