# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 1e-4 /1.4  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=7e-6)
# runtime settings
total_epochs = 40