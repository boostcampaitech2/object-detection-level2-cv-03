
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='P-stage2-detection',
                name='CBNet_default',
                entity='boostcampaitech2-object-detection-level2-cv-03')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    save_best='bbox_mAP',
    metric=['bbox']
)