_base_ = ['../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py']

MODEL_NAME ='centernet2_RFP_DCN_resNext101_64x4d_1x_coco'
CV_VER = 1

model = dict(
     backbone=dict(
        type='DetectoRS_ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        output_img=True,
        #stage_with_sac=(False, True, True, True),
        #sac=dict(type='SAC', use_deform=True),
        #conv_cfg=dict(type='ConvAWS'),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    neck=dict(
        _delete_=True,
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=64,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            #conv_cfg=dict(type='ConvAWS'),
            # sac=dict(type='SAC', use_deform=True),
            # stage_with_sac=(False, True, True, True),
            dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
            pretrained='open-mmlab://resnext101_64x4d',
            style='pytorch')),
    rpn_head=dict(
        num_classes=10,
        use_deformable=True
    )
)

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
        type='RandomRotate90',p=0.5
    )
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CustomCutmix',p=0.5,T=1),
    #dict(type='Resize', img_scale=[(1024, 512),(1024,768)], keep_ratio=True,multiscale_mode= 'range'),
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
    samples_per_gpu=2,
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


lr_config = dict(step=[27,33,39,45])
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
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='p-stage_object_detection',
                name=f'{MODEL_NAME}',
                #entity='boostcampaitech2-object-detection-level2-cv-03'
                ))])




