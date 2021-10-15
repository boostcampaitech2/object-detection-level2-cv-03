# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 1e-4  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=7e-6)
# # runtime settings
total_epochs = 300


# optimizer = dict(
#     type='SGD',
#     lr=1e-4,
#     momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)

# lr_config = dict(
#     # _delete_=True,
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=15,
#     min_lr_ratio=0.05)
# runner = dict(type='EpochBasedRunner', max_epochs=300)