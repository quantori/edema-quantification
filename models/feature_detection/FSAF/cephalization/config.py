model = dict(
    type='FSAF',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'),
        groups=64,
        base_width=4),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='FSAFHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            type='IoULoss', eps=1e-06, loss_weight=1.0, reduction='none'),
        reg_decoded_bbox=True),
    train_cfg=dict(
        assigner=dict(
            type='CenterRegionAssigner',
            pos_scale=0.2,
            neg_scale=0.2,
            min_pos_iof=0.01),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = 'data/coco_cephalization'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(1333, 800),
        multiscale_mode='range',
        ratio_range=[0.75, 1],
        keep_ratio=True,
        bbox_clip_border=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3,
        bbox_clip_border=True),
    dict(type='RandomFlip', direction='horizontal', flip_ratio=0.5),
    dict(type='Rotate', level=1, max_rotate_angle=20, prob=0.2),
    dict(type='Translate', level=1, max_translate_offset=80, prob=0.2),
    dict(type='EqualizeTransform', prob=0.2),
    dict(type='BrightnessTransform', level=1, prob=0.2),
    dict(type='ContrastTransform', level=1, prob=0.2),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco_cephalization/train/labels.json',
        img_prefix='data/coco_cephalization/train/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('Cephalization', ),
        filter_empty_gt=False),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco_cephalization/test/labels.json',
        img_prefix='data/coco_cephalization/test/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Cephalization', )),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco_cephalization/test/labels.json',
        img_prefix='data/coco_cephalization/test/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Cephalization', )))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(
    type='Adam', lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = None
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MlflowLoggerHook',
            exp_name='Default',
            log_model=True,
            interval=1,
            params=dict(),
            ignore_last=False)
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
classes = ('Cephalization', )
seed = 11
total_epochs = 100
work_dir = 'models/feature_detection/fsaf_cephalization'
auto_resume = False
gpu_ids = [0]
