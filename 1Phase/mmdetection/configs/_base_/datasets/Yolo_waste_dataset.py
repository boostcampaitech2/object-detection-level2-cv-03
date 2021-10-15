# dataset settings
# data_root = 'data/coco/'

data_root = '/opt/ml/detection/dataset/'
dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

multi_scale = (640, 640)
# multi_scale = [[x,x] for x in range(512, 1024+1, 32)]

train_pipeline = [
    dict(type='Mosaic', img_scale=multi_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-multi_scale[0] // 2, -multi_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=multi_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        ann_file=data_root + 'cv_train1.json',
        classes=classes,
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
    dynamic_scale=multi_scale)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=multi_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=multi_scale, pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'cv_val1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'cv_val1.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
