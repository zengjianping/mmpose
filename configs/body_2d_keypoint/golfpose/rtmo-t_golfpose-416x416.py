_base_ = ['../../_base_/default_runtime.py']

# common setting
num_keypoints = 30

# runtime
train_batch_size = 16
val_batch_size = 4
train_cfg = dict(max_epochs=60, val_interval=2, dynamic_intervals=[(54, 1)])

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=3))

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
        T_max=25,
        end=25,
        by_epoch=True,
        convert_to_iter_based=True),
    # this scheduler is used to increase the lr from 2e-4 to 5e-4
    dict(type='ConstantLR', by_epoch=True, factor=2.5, begin=25, end=26),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=26,
        T_max=30,
        end=55,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=55, end=60),
]

# data
input_size = (416, 416)
metafile = 'configs/_base_/datasets/golfpose.py'
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
dataset_type = 'GolfPoseDataset'
data_mode = 'bottomup'
data_root = 'data/'
backend_args = None

# mapping
golfpose_golfpose = [(i, i) for i in range(30)]
golfpose_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=golfpose_golfpose)

coco_golfpose = [(i, i) for i in range(17)] + [(17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)] + [(100, 26), (121, 27)]
coco_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=coco_golfpose)

halpe_golfpose = [(i, i) for i in range(26)] + [(103, 26), (124, 27)]
halpe_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=halpe_golfpose)

aic_golfpose = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7),
                (5, 9), (6, 12), (7, 14), (8, 16), (9, 11), (10, 13), (11, 15),
                (12, 17), (13, 18)]
aic_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=aic_golfpose)

crowdpose_golfpose = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                      (7, 12), (8, 13), (9, 14), (10, 15), (11, 16), (12, 17),
                      (13, 18)]
crowdpose_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=crowdpose_golfpose)

mpii_golfpose = [
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
mpii_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=mpii_golfpose)

jhmdb_golfpose = [
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
jhmdb_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=jhmdb_golfpose)

ochuman_golfpose = [(i, i) for i in range(17)]
ochuman_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=ochuman_golfpose)

posetrack_golfpose = [
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
posetrack_golfpose_converter = dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=posetrack_golfpose)

# train datasets
dataset_ezgolf = dict(
    type='GolfPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240418/annotations/golfpose/train.json',
    data_prefix=dict(img='ezgolf/task_20240418/images'),
    pipeline=[golfpose_golfpose_converter],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_golfdb = dict(
    type='GolfPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='golfdb/annotations/golfpose/train.json',
    data_prefix=dict(img='golfdb/images'),
    pipeline=[golfpose_golfpose_converter],
    #indices=1000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

dataset_coco = dict(
    type='CocoWholeBodyDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=[coco_golfpose_converter],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_v1.json',
    data_prefix=dict(img='halpe/hico_20160224_det/images/train2015'),
    pipeline=[halpe_golfpose_converter],
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='pose/ai_challenge/ai_challenger_keypoint'
                     '_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[aic_golfpose_converter],
)

dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='crowdpose/images/'),
    pipeline=[crowdpose_golfpose_converter],
)

dataset_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='mpii/images/'),
    pipeline=[mpii_golfpose_converter],
)

dataset_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_train.json',
    data_prefix=dict(img='jhmdb/'),
    pipeline=[jhmdb_golfpose_converter],
)

dataset_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_train.json',
    data_prefix=dict(img='posetrack18/'),
    pipeline=[posetrack_golfpose_converter],
)

train_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file=metafile),
    datasets=[
        dataset_ezgolf,
        dataset_golfdb,
        dataset_coco,
        dataset_halpe,
        #dataset_aic,
        #dataset_crowdpose,
        #dataset_mpii,
        #dataset_jhmdb,
        #dataset_posetrack,
    ],
    sample_ratio_factor=[2, 1, 0.5, 0.5],
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
    type='GolfPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ezgolf/task_20240418/annotations/golfpose/val.json',
    data_prefix=dict(img='ezgolf/task_20240418/images'),
    dataset_name='ezgolf',
    pipeline=[golfpose_golfpose_converter],
    #indices=50,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

val_golfdb = dict(
    type='GolfPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='golfdb/annotations/golfpose/val.json',
    data_prefix=dict(img='golfdb/images'),
    dataset_name='golfdb',
    pipeline=[golfpose_golfpose_converter],
    #indices=50,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
)

val_coco = dict(
    type='CocoWholeBodyDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_val_v1.0.json',
    data_prefix=dict(img='coco/val2017/'),
    pipeline=[coco_golfpose_converter],
)

val_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_val_v1.json',
    data_prefix=dict(img='coco/val2017/'),
    pipeline=[halpe_golfpose_converter],
)

val_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_val.json',
    data_prefix=dict(
        img='aic/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'),
    pipeline=[aic_golfpose_converter],
)

val_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_test.json',
    data_prefix=dict(img='crowdpose/images/'),
    pipeline=[crowdpose_golfpose_converter],
)

val_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_val.json',
    data_prefix=dict(img='mpii/images/'),
    pipeline=[mpii_golfpose_converter],
)

val_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_test.json',
    data_prefix=dict(img='jhmdb/'),
    pipeline=[jhmdb_golfpose_converter],
)

val_ochuman = dict(
    type='OCHumanDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ochuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json',
    data_prefix=dict(img='ochuman/images/'),
    pipeline=[ochuman_golfpose_converter],
)

val_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_val.json',
    data_prefix=dict(img='posetrack18/'),
    pipeline=[posetrack_golfpose_converter],
)

val_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file=metafile),
    datasets=[
        val_ezgolf,
        val_golfdb,
        #val_coco,
        val_halpe,
        #val_aic,
        #val_crowdpose,
        #val_mpii,
        #val_jhmdb,
        #val_posetrack,
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
val_evaluator_golfdb = dict(
    type='CocoMetric',
    ann_file=data_root + val_golfdb['ann_file'],
    score_mode='bbox',
    nms_mode='none',
    format_only=False,
    prefix='golfdb'
)
val_evaluator_coco = dict(
    type='CocoMetric',
    ann_file=data_root + val_coco['ann_file'],
    score_mode='bbox',
    nms_mode='none',
    format_only=False,
    gt_converter=coco_golfpose_converter,
    prefix='coco'
)
val_evaluator_halpe = dict(
    type='CocoMetric',
    ann_file=data_root + val_halpe['ann_file'],
    score_mode='bbox',
    nms_mode='none',
    format_only=False,
    gt_converter=halpe_golfpose_converter,
    prefix='halpe'
)
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    datasets=[val_ezgolf, val_golfdb, val_halpe],
    metrics=[val_evaluator_ezgolf, val_evaluator_golfdb, val_evaluator_halpe],
)
test_evaluator = val_evaluator

# hooks
custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=3,
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
            dynamic_k_indicator='oks',
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
