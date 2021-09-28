_base_ = './vfnet_r50_fpn.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(dcn_on_last_conv=True))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

alb_transform = [
        dict(
        type='OneOf',
        transforms=[
            dict(
                type='GaussNoise',
                p=1.0),
            dict(
                type='GaussianBlur',
                p=1.0),
            dict(
                type='Blur',
                p=1.0),
        ],
        p=0.1),
        dict(
        type='OneOf',
        transforms=[
            dict(
                type='CLAHE',
                p=1.0),
            dict(
                type='RandomGamma',
                p=1.0),
            dict(
                type='HueSaturationValue',
                p=1.0),
            dict(
                type='ChannelDropout',
                p=1.0),
            dict(
                type='ChannelShuffle',
                p=1.0),
            dict(
                type='RGBShift',
                p=1.0)
        ],
        p=0.1),
        dict(
        type='OneOf',
        transforms=[
            # dict(
            #     type='VerticalFlip',
            #     p=1.0),
            # dict(
            #     type='HorizontalFlip',
            #     p=1.0),
            dict(
                type='ShiftScaleRotate',
                p=1.0),
            dict(
                type='Rotate',
                p=1.0),
        ],
        p=0.1)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1024, 512), (1024, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=alb_transform,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
                'img': 'image',
                'gt_bboxes': 'bboxes'
        },
    update_pad_shape=False,
    skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=40)