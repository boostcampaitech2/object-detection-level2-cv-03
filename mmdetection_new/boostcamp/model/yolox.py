_base_=['../configs/yolox/yolox_x_8x8_300e_coco.py']

MODEL_NAME ='yolox_x_8x8_300e_coco_batch2'
CV_VER = 1
LR = 0.01
BATCH_SIZE=2
ORIGIN_BATCH_SIZE=8
BATCH_RATIO = BATCH_SIZE/ORIGIN_BATCH_SIZE

model = dict(
    bbox_head=dict(
        num_classes=10
    )
)

img_scale = (1024,1024)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


# albu_train_transforms = [
#     dict(
#         type='CLAHE',
#         tile_grid_size=(8,8),
#         p=0.5
#     ),
#     dict(
#         type='PiecewiseAffine',p=0.2
#     ),
#     dict(
#         type='RandomRotate90',p=0.5
#     ),
# ]

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_masks': 'masks',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + f'kfold/cv_val{CV_VER}.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=False),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= img_scale,
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
    train=train_dataset,
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + f'kfold/cv_train{CV_VER}.json',
    #     img_prefix=data_root,
    #     classes=classes,
    #     pipeline=train_pipeline
    #     ),
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


optimizer=dict(lr=LR*BATCH_RATIO)
runner = dict(type='EpochBasedRunner', max_epochs=600)

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




