_base_ = ['/opt/ml/detection/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']

CV_VER = 1
MODEL_NAME = 'epoch_test_prototype'
model=dict(
   roi_head=dict(
       bbox_head=dict(
           num_classes=10
       )
)
)
work_dir = f'./boostcamp/work_dirs/{MODEL_NAME}'

dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(32, 32), (32, 32)], keep_ratio=True,multiscale_mode= 'range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= [(32, 32)],
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
    samples_per_gpu=4,
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


log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='p-stage_object_detection',
                name=f'{MODEL_NAME}',
                #entity='boostcampaitech2-object-detection-level2-cv-03'
                ))])