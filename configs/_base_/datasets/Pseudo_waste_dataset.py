dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/Augmented/Pseudos/5495pse_B_B+pse_Univ_LB674/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

size_min = 512
# size_max = 1024
size_max = 1024

multi_scale_list = []
for i in range(size_min, size_max+1,32):
    multi_scale_list.append((i,i))

multi_scale_list_tta3 = []
for i in range(size_min, size_max+1,256):
    multi_scale_list_tta3.append((i,i))

alb_transform = [
    dict(
        type='VerticalFlip',
        p=0.15),
    dict(
        type='HorizontalFlip',
        p=0.3),
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
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            # dict(
            #     type='CLAHE',
            #     p=1.0),
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
            dict(
                type='ShiftScaleRotate',
                p=1.0),
            dict(
                type='RandomRotate90',
                p=1.0),
        ],
        p=0.1)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=multi_scale_list, multiscale_mode='value', keep_ratio=True),
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
        img_scale=multi_scale_list_tta3,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=multi_scale_list_tta3, multiscale_mode='value', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'full_pseudo_cv_train1.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'genuine/cv_val1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'cv_val1.json',
        img_prefix=data_root,
        pipeline=test_pipeline))