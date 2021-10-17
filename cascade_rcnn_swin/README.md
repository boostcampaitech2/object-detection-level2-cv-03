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
- trains on a small & medium biased dataset
- uses expanded multi scales [800 ~ 1408]
- Anchor_ratios and anchor_scales focused on small and medium objects  


## Requirements
**Libraries**
- Ubuntu 18.04 LTS
- Python 3.7.5
- PyTorch 1.7.1 <=
- mmdet 2.17.0  

**Hardware**
- GPU: 1 x NVIDIA Tesla V100 32G

## Train Models (GPU needed)
On a single GPU
```
python tools/train.py [path to swin*_cascade*.py]
```

On multiple GPUs
```
tools/dist_train.sh [path to swin*_cascade*.py] [number of GPUs]  
```  
## Dataset
**Default datatset**  
<img width="512" alt="image" src="https://user-images.githubusercontent.com/30382262/137625618-39656c65-ed13-42f0-8659-a3d7cd45f60c.jpg">  
[네이버 커넥트재단 - 재활용 쓰레기 데이터셋 / CC BY 2.0]

**small & medium biased dataset**  
<img width="512" alt="image" src="https://user-images.githubusercontent.com/30382262/137625208-37fd84a5-fccb-42cb-9947-1660082fcd9e.png">  
[네이버 커넥트재단 - 재활용 쓰레기 데이터셋 / CC BY 2.0]
