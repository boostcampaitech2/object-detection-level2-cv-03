_base_ = [
    '../../datasets/Yolo_waste_dataset.py',
    '../../schedules/Yolo_waste_schedule.py', '../../Yolo_waste_runtime.py',
]


pretrained='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth'
# model settings
model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4),
    bbox_head=dict(
        type='YOLOXHead', num_classes=10, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

img_scale = (640,640)
resume_from = None
interval = 10

custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(14, 26),
        img_scale=img_scale,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]
log_config = dict(interval=10)
