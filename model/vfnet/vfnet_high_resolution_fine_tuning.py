_base_ = ['vfnet_swin.py']

root = '/opt/ml/detection/mmdetection/boostcamp/work_dirs/'
file = 'no_TTA_pseudo_vfnet_swinB_pafpn/1/best_bbox_mAP_50_epoch_40.pth'
resume_from = root+file

MODEL_NAME ='no_TTA_pseudo_vfnet_swinB_pafpn'
CV_VER = 1

albu_train_transforms = [
    dict(
        type='RandomRotate90',p=0.5
    )
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=[(2048,2048)],
        multiscale_mode='value',
        keep_ratio=True
    ),
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

data=dict(
    samples_per_gpu=1,
    train=dict(pipeline=train_pipeline))
optimizer = dict(lr=0.00001)
checkpoint_config = dict(interval=1)
lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    periods=[20,20,12],
    restart_weights=[1.,1.,1.],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-7,
    )

runner = dict(type='EpochBasedRunner', max_epochs=52)

work_dir = root + f'{MODEL_NAME}/r_fine'

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='p-stage_object_detection',
                name='r_fine_'+ MODEL_NAME,
                #entity='boostcampaitech2-object-detection-level2-cv-03'
                ))])


