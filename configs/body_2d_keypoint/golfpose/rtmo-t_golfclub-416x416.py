_base_ = ['../../_base_/default_runtime.py']

# common setting
num_keypoints = 2

# runtime
train_batch_size = 16
val_batch_size = 4

# runtime
train_cfg = dict(max_epochs=80, val_interval=2, dynamic_intervals=[(70, 2)])

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=10))

optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.002, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({'neck.encoder': dict(lr_mult=0.05)})),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=10,
        T_max=30,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True),
    # this scheduler is used to increase the lr from 2e-4 to 5e-4
    dict(type='ConstantLR', by_epoch=True, factor=2.5, begin=30, end=31),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=31,
        T_max=40,
        end=70,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=70, end=80),
]

# data
input_size = (416, 416)
metafile = 'configs/_base_/datasets/golfclub.py'
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(416, 416),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(416, 416),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(416, 416),
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('dataset_name', 'id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale', 'raw_ann_info'))
]

# data settings
dataset_type = 'GolfClubDataset'
data_mode = 'bottomup'
data_root = 'data/'
backend_args = None

# train datasets
dataset_ezgolf = dict(
    type='GolfClubDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240418/annotations/golfclub/train.json',
    data_prefix=dict(img='ezgolf/task_20240418/images'),
    pipeline=[],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_ezgolf_t2 = dict(
    type='GolfClubDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240628/annotations/golfclub/train.json',
    data_prefix=dict(img='ezgolf/task_20240628/images'),
    pipeline=[],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_golfdb = dict(
    type='GolfClubDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='golfdb/annotations/golfclub/train.json',
    data_prefix=dict(img='golfdb/images'),
    pipeline=[],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

train_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file=metafile),
    datasets=[
        dataset_ezgolf,
        dataset_ezgolf_t2,
        dataset_golfdb,
    ],
    sample_ratio_factor=[1, 2, 1],
    test_mode=False,
    pipeline=train_pipeline_stage1
)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    #num_batch_per_epoch=500,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

# val datasets
val_ezgolf = dict(
    type='GolfClubDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240418/annotations/golfclub/val.json',
    data_prefix=dict(img='ezgolf/task_20240418/images'),
    dataset_name='ezgolf',
    pipeline=[],
    #indices=50,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

val_ezgolf_t2 = dict(
    type='GolfClubDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240628/annotations/golfclub/val.json',
    data_prefix=dict(img='ezgolf/task_20240628/images'),
    dataset_name='ezgolf_t2',
    pipeline=[],
    #indices=50,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

val_golfdb = dict(
    type='GolfClubDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='golfdb/annotations/golfclub/val.json',
    data_prefix=dict(img='golfdb/images'),
    dataset_name='golfdb',
    pipeline=[],
    #indices=50,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

val_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file=metafile),
    datasets=[
        val_ezgolf,
        val_ezgolf_t2,
        val_golfdb,
    ],
    test_mode=True,
    pipeline=val_pipeline,
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    #num_batch_per_epoch=50,
    persistent_workers=True,
    pin_memory=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=val_dataset
)
test_dataloader = val_dataloader

# evaluators
val_evaluator_ezgolf = dict(
    type='CocoMetric',
    ann_file=data_root + val_ezgolf['ann_file'],
    score_mode='bbox',
    nms_mode='none',
    format_only=False,
    prefix='ezgolf'
)
val_evaluator_ezgolf_t2 = dict(
    type='CocoMetric',
    ann_file=data_root + val_ezgolf_t2['ann_file'],
    score_mode='bbox',
    nms_mode='none',
    format_only=False,
    prefix='ezgolf_t2'
)
val_evaluator_golfdb = dict(
    type='CocoMetric',
    ann_file=data_root + val_golfdb['ann_file'],
    score_mode='bbox',
    nms_mode='none',
    format_only=False,
    prefix='golfdb'
)
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    datasets=[val_ezgolf, val_ezgolf_t2, val_golfdb],
    metrics=[val_evaluator_ezgolf, val_evaluator_ezgolf_t2, val_evaluator_golfdb],
)
test_evaluator = val_evaluator

# hooks
custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=10,
        new_train_dataset=train_dataset,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(
        type='RTMOModeSwitchHook',
        epoch_attributes={
            280: {
                'proxy_target_cc': True,
                'loss_mle.loss_weight': 5.0,
                'loss_oks.loss_weight': 10.0
            },
        },
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

# model
widen_factor = 0.375
deepen_factor = 0.33

model = dict(
    type='BottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(320, 640),
                size_divisor=32,
                interval=1),
        ]),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/'
            'yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_'
            '20211124_171234-b4047906.pth',
            prefix='backbone.',
        )),
    neck=dict(
        type='HybridEncoder',
        in_channels=[96, 192, 384],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1, 2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=192,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2)),
    head=dict(
        type='RTMOHead',
        num_keypoints=num_keypoints,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            cls_feat_channels=256,
            channels_per_group=36,
            pose_vec_channels=192,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='iou',
            iou_calculator=dict(type='BBoxOverlaps2D'),
            oks_calculator=dict(type='PoseOKS', metainfo=metafile),
            use_keypoints_for_center=True),
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]),
        dcc_cfg=dict(
            in_channels=192,
            feat_channels=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_cfg=dict(
                s=128,
                expansion_factor=2,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn='SiLU',
                pos_enc='add')),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo=metafile,
            loss_weight=30.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    ))
