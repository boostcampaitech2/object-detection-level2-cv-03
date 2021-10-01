dataset_type = 'CocoDataset'
# data_root = '/home/hci/Videos/datasetBGAug/'
data_root = '/home/hci/Videos/archive/dataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

min_size = 416
# imsize = 1024 - 32*4
imsize = 512

multi_scale_dict = []
for i in range(min_size,imsize+1,32):
    multi_scale_dict.append(dict(type='Resize', height=i, width=i))
'''
multi_scale_dict = []
for w in range(512,imsize+1,32):
    for h in range(512,imsize+1,32):
        multi_scale_dict.append(dict(type='Resize', height=h, width=w))
'''
alb_transform = [
    # dict(
    #     type='OneOf',
    #     transforms=multi_scale_dict,
    #     p=1.0),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='GaussNoise',
    #             p=1.0),
    #         dict(
    #             type='GaussianBlur',
    #             p=1.0),
    #         dict(
    #             type='Blur',
    #             p=1.0)
    #     ],
    #     p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='CLAHE',
    #             p=1.0),
    #         dict(
    #             type='RandomGamma',
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             p=1.0),
    #         dict(
    #             type='ChannelDropout',
    #             p=1.0),
    #         dict(
    #             type='ChannelShuffle',
    #             p=1.0),
    #         dict(
    #             type='RGBShift',
    #             p=1.0)
    #     ],
    #     p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         # dict(
    #         #     type='VerticalFlip',
    #         #     p=1.0),
    #         # dict(
    #         #     type='HorizontalFlip',
    #         #     p=1.0),
    #         dict(
    #             type='ShiftScaleRotate',
    #             p=1.0),
    #         dict(
    #             type='Rotate',
    #             p=1.0),
    #     ],
    #     p=0.1)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(imsize, imsize)),
    # dict(type='Resize', img_scale=[(1024, 1024), (1024, 512)]),
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
    # dict(type='CutOut', n_holes=4, cutout_shape=(30, 30)),
    # dict(type='CutOut', n_holes=(0,5), cutout_shape=(32,32)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(imsize, imsize),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train_v1.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val_v1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val_v1.json',
        img_prefix=data_root,
        pipeline=test_pipeline))