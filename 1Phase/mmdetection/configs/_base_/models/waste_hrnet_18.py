_base_ = [
    # '../datasets/waste_dataset.py',
    # '../schedules/waste_schedule.py', '../waste_runtime.py',
    # '/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/configs/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco.py'
    './waste_hrnet_32.py'
]


model = dict(
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18')),
    neck=dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256))



data = dict(
    samples_per_gpu = 8
)