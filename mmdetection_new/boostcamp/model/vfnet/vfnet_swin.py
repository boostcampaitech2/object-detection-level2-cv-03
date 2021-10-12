_base_=['/opt/ml/detection/mmdetection/configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py']

MODEL_NAME ='no_TTA_pseudo_vfnet_swinB_pafpn'
CV_VER = 1
BATCH_SIZE=2


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'
model = dict(
     backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    bbox_head=dict(
        num_classes=10
    ),
    neck=dict(
        type='PAFPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5))
    

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

pseudo_root= '/opt/ml/detection/dataset/pseudo_dataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
## Non-rigid Albumentation
albu_train_transforms = [
    dict(
        type='RandomRotate90',p=0.5
    ),
    dict(
        type='CLAHE',
        tile_grid_size=(8,8),
        p=0.5
    )
]

###

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='CustomCutmix',p=0.5,T=1,
                dataset_type=dataset_type,
                ann_file = pseudo_root+'pseudo_modified.json',
                img_prefix=pseudo_root,
                classes=classes)],
            [
            dict(type='CustomCutmix',p=0.5,T=1,
                dataset_type=dataset_type,
                ann_file =data_root + f'kfold/cv_train{CV_VER}.json',
                img_prefix=data_root,
                classes=classes)]]
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333),(1024, 1333)],
                multiscale_mode='range',
                keep_ratio=True)],
                [
            dict(
                type='Resize',
                img_scale=[(400, 1333), (800, 1333)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=True),
            dict(
                type='Resize',
                img_scale=[(480, 1333),(1024, 1333)],
                multiscale_mode='range',
                override=True,
                keep_ratio=True)],
                  ]),
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
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline= [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= [(1024, 1024)],
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
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))



optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(step=[27,33,39,45],gamma=0.2)
runner = dict(type='EpochBasedRunner', max_epochs=48)


# 공통 settings
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
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='p-stage_object_detection',
                name=MODEL_NAME,
                #entity='boostcampaitech2-object-detection-level2-cv-03'
                ))])




