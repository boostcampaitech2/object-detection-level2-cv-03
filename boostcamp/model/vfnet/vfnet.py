_base_=['../configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py']

MODEL_NAME ='no-rigid-aug_vfnet_r50_pafpn_mdconv_c3-c5_mstrain_2x_coco'
CV_VER = 1
LR = 0.001
BATCH_SIZE=8
ORIGIN_BATCH_SIZE=2
BATCH_RATIO = BATCH_SIZE/ORIGIN_BATCH_SIZE

model = dict(
    # backbone=dict(
        # type='ResNeSt',
        # stem_channels=64,
        # depth=50,
        # radix=2,
        # reduction_factor=4,
        # avg_down_stride=True,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_eval=False,
        # style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnest50'))
    bbox_head=dict(
        num_classes=10
    ),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))


# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

## Non-rigid Albumentation
albu_train_transforms = [
    dict(
        type='CLAHE',
        tile_grid_size=(8,8),
        p=0.5
    ),
    # Not yet applied
    dict(
        type='Affine',p=0.5,
        scale=[0.5,1.5],
        shear=[-45,45]
    ),
    dict(
        type='PiecewiseAffine',p=0.5
    ),
    dict(
        type='RandomRotate90',p=0.5
    ),
]

###
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=[(1024, 512),(1024,768)], keep_ratio=True,multiscale_mode= 'range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333)#, (800, 1333)
                           ],
                multiscale_mode='value',
                keep_ratio=True)],
                [
            dict(
                type='Resize',
                img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=True),
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333)#, (800, 1333)
                            ],
                multiscale_mode='value',
                override=True,
                keep_ratio=True)],
                [
            dict(
                type='Albu',
                transforms=albu_train_transforms,
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap={
                    'img': 'image',
                    'gt_masks': 'masks',
                    'gt_bboxes': 'bboxes'
                },
                update_pad_shape=False,
                skip_img_without_anno=True)]
                  ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= [(512,512),(840,840),(1024, 1024)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + f'kfold/cv_train{CV_VER}.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + f'kfold/cv_val{CV_VER}.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))

# optimizer settings
optimizer = dict(lr=LR*BATCH_RATIO)

# learning rate scheduler settings
lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    periods=[20,20,20],
    restart_weights=[1.,0.5,0.5],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
    )

# runner settigns
runner = dict(
    _delete_=True,
    type='EpochBasedRunner', max_epochs=60)


## 공통 settings
evaluation = dict(
    interval=1,
    save_best='bbox_mAP_50',
    metric=['bbox']
)
seed = 1010
gpu_ids = [0]
work_dir = f'./boostcamp/work_dirs/{MODEL_NAME}/{CV_VER}'

log_config = dict(
    _delete_=True,
    interval=10,
    hooks=[
        #dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='p-stage_object_detection',
                name=f'{MODEL_NAME}',
                #entity='boostcampaitech2-object-detection-level2-cv-03'
                ))])

# lr_config = dict(
#     _delete_=True,
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     gamma=0.2,
#     step=[16,22,28,34])




