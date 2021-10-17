# object-detection-level2-cv-03
## Contents
```
.
└── configs
    └── _base_
        ├── Swin_waste_runtime.py
        ├── Yolo_waste_runtime.py
        ├── datasets
        │   ├── Pseudo_waste_dataset.py
        │   ├── Swin_waste_dataset.py
        │   ├── Yolo_waste_dataset.py
        │   └── waste_dataset.py
        ├── default_runtime.py
        ├── models
        │   ├── Swin
        │   │   ├── swinB_cascade.py
        │   │   └── swinS_cascade_for_smallObjs.py
        │   ├── VFNet
        │   │   ├── waste_vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x.py
        │   │   └── waste_vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x.py
        │   └── YoloX
        │       ├── yolox_l.py
        │       └── yolox_x.py
        ├── schedules
        │   ├── Yolo_waste_schedule.py
        │   └── waste_schedule.py
        └── waste_runtime.py
```
