# Usage
1. custom_cutmix_aug.py와 augmixations을 mmdetection/mmdet/datasets/pipelines 내부에 저장
<br>
<br>
2. mmdet/datasets/pipelines/__init__.py 내부에 from .custom_cutmix_aug import CustomCutmix 한 후 __all__ 리스트에 CustomCutmix 추가

ex)
```python3
from .custom_cutmix_aug import CustomCutmix

__all__ = [
    ...,
    'CustomCutmix']
```
<br>
<br>
3. pipeline내부에 CustomCutmix 추가

ex)
```python3
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CustomCutmix',p=0.5),
    dict(type='Resize', img_scale=[(384, 512), (512, 512)], keep_ratio=True,multiscale_mode= 'range'),
    dict(type='RandomFlip', flip_ratio=0.5),       
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
``` 

| 주의 : LoadAnnotations 앞에 추가해줘야 한다.
