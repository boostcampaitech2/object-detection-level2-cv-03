expr_name = 'yolox_x_adamw'
# expr_name = 'swinL7_992'

dist_params = dict(backend='nccl')

runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=30)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='P-stage2-detection-Augs',
                name=expr_name)
        )
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(save_best='bbox_mAP', metric=['bbox'])
work_dir = './work_dirs/' + expr_name
gpu_ids = range(0, 1)