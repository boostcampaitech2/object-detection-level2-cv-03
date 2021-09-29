_base_ = [
    '../datasets/waste_dataset.py',
    '../schedules/waste_schedule.py', '../waste_runtime.py',
    '/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/configs/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco.py'
]

data = dict(
    samples_per_gpu = 16
)