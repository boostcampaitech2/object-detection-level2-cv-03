_base_ = [
    '/opt/ml/detection/mmdetection_update/mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py',
    '/opt/ml/detection/mmdetection_update/mmdetection/configs/_base_/datasets/custom_dataset.py',
    '/opt/ml/detection/mmdetection_update/mmdetection/configs/_base_/schedules/schedule_3x.py',
    '/opt/ml/detection/mmdetection_update/mmdetection/configs/_base_/custom_runtime.py'
]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

runner = dict(type='EpochBasedRunner', max_epochs=40)