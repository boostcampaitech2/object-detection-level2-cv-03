This repo contains the 2nd solutions on BoostCamp AI_Tech (2nd term) object detection competetion.  

# object-detection-level2-cv-03
## Contents
```
.
└── _base_
    ├── datasets
    ├── models
    │   └── Swin
    │       ├── swinB_cascade.py
    │       └── swinS_cascade_for_smallObjs.py
    └── schedules
```
`swinB_cascade.py`: It trains model on default dataset
- mAP50 0.703
- trains on a default dataset (waste_coco: imsize is 1024x1024)
- uses multi-scaled images [512 ~ 1024]
- default anchor ratios
- and default settings  

`swinS_cascade_for_smallObjs.py`: Focus on the small and medium objects.
- trains on a small / medium biased dataset
- uses expanded multi scales [800 ~ 1408]
- Anchor ratios / scales focused on small and medium objects  


## Requirements
- Ubuntu 18.04 LTS
- Python 3.7.5
- PyTorch 1.7.1 <=
- mmdet 2.17.0  

# Hardware
- GPU: 1 x NVIDIA Tesla V100 32G

## Train Models (GPU needed)
On a single GPU
```
python tools/train.py [path to swin*_cascade*.py]
```

On multiple GPUs
```
tools/dist_train.sh [path to swin*_cascade*.py] [number of GPUs]  


