_base_ = ['/opt/ml/detection/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']

CV_VER = 1
MODEL_NAME = 'motion_blur_aug_test'
model=dict(
   roi_head=dict(
       bbox_head=dict(
           num_classes=10
       )
)
)
work_dir = f'./boostcamp/work_dirs/aug_test/motion_blur_aug_test/'

dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# albu_train_transforms = [
#     dict(
#         type='Affine',p=0.5,
#         scale=[0.5,1.5],
#         shear=[-30,30]
#     )
# ]

albu_train_transforms = [
    dict(
        type='MotionBlur',p=0.5
    )
]


# # albu_train_transforms = [
# #     dict(
# #         type='PiecewiseAffine',p=0.5,
# #         scale=(0.03, 0.08)
# #     )
# # ]

# albu_train_transforms = [
#     dict(
#         type='CLAHE',
#         tile_grid_size=(8,8),
#         p=0.5
#     ),
# ]

## pipeline settings
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='CustomCutmix',p=0.5),
    #dict(type='Mosaic', img_scale=(512,512), pad_val=114.0),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    dict(type='Resize', img_scale=[(384, 512), (512, 512)], keep_ratio=True,multiscale_mode= 'range'),
    dict(type='RandomFlip', flip_ratio=0.5),      
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

test_pipeline = [
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




lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)


evaluation = dict(
    interval=1,
    save_best='bbox_mAP_50',
    metric=['bbox']
)

seed = 1010
gpu_ids = [0]


# log_config = dict(
#     _delete_=True,
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(
#             type='WandbLoggerHook',
#             init_kwargs=dict(
#                 project='p-stage_object_detection',
#                 name=f'{MODEL_NAME}',
#                 #entity='boostcampaitech2-object-detection-level2-cv-03'
#                 ))])