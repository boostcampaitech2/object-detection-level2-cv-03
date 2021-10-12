_base_=['../configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py']

MODEL_NAME ='no-rigid-aug_cascade_rcnn_r50_fpn_1x_coco'
CV_VER = 1
LR = 0.04
BATCH_SIZE=8
ORIGIN_BATCH_SIZE=8
BATCH_RATIO = BATCH_SIZE/ORIGIN_BATCH_SIZE

model= dict(
   rpn_head=dict(
       anchor_generator=dict(
           scales=[2,4,8,16],
           ratios=[0.33,0.5,1.0,2.0,3.0]
       ),
       loss_cls=dict(
           _delete_=True,
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(
            _delete_=True,
            type='GIoULoss',
            loss_weight=1.5),
   ),

   roi_head=dict(
        bbox_head=[
            dict(
              loss_cls=dict(
                  _delete_=True,
                    type='VarifocalLoss',
                    alpha=0.75,
                    gamma=2.0,
                    iou_weighted=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    _delete_=True,
                    type='GIoULoss', loss_weight=1.5)),
            dict(
               loss_cls=dict(
                   _delete_=True,
                    type='VarifocalLoss',
                    use_sigmoid=True,
                    alpha=0.75,
                    gamma=2.0,
                    iou_weighted=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    _delete_=True,
                    type='GIoULoss', loss_weight=1.5)),
            dict(
                loss_cls=dict(
                    _delete_=True,
                    type='VarifocalLoss',
                    use_sigmoid=True,
                    alpha=0.75,
                    gamma=2.0,
                    iou_weighted=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    _delete_=True,
                    type='GIoULoss', loss_weight=1.5))
        ]),
    test_cfg=dict(
        rpn=dict(
            nms=dict(type='soft_nms', iou_threshold=0.7)),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5)))
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
        type='CLAHE',
        tile_grid_size=(8,8),
        p=0.5
    ),
    # Not yet applied
    dict(
        type='Affine',p=0.5,
        scale=[0.5,1.5],
        shear=[-30,30]
    ),
    dict(
        type='PiecewiseAffine',p=0.2
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


optimizer = dict(lr=LR*BATCH_RATIO)

lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    periods=[12,12,12],
    restart_weights=[1.,0.5,0.5],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5,
    )
# lr_config = dict(
#     _delete_=True,
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     gamma=0.2,
#     step=[16,22,28,34])

runner = dict(
    _delete_=True,
    type='EpochBasedRunner', max_epochs=36)


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