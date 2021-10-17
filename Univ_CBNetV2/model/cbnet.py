_base_ = [
    '/opt/ml/detection/UniverseNet/configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py',
    './univ_dataset.py',
    './univ_schedule.py',
    './univ_runtime.py'
]

model = dict(
    backbone=dict(
        type='CBRes2Net',
        cb_del_stages=1,
        cb_inplanes=[64, 256, 512, 1024, 2048],
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(type='CBFPN'),
    test_cfg=dict(rcnn=dict(score_thr=0.001, nms=dict(type='soft_nms'))))
    

