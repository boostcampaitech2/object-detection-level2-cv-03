<h1 align="center">
<p>Cascade R-CNN
</h1>

```bash
cascade_rcnn_swinB_anchor
├── _base_
    ├── custom_dataset.py
    ├── schedule_3x.py
    ├── custom_runtime.py
    └── cascade_rcnn_swin_fpn.py
└── model
    └── cascade_rcnn_swinB_anchor.py
```
## 파일 설명:

`custom_dataset.py`: Dataset 관련 custom config 파일

`schedule_3x.py`: Cosine Annealing을 적용한 schedule 관련 custom config 파일

`custom_runtime.py`: log와 evaluation 관련 custom config 파일

`cascade_rcnn_swin_fpn.py`: Cascade R-CNN에 Swin Transformer backbone을 도입한 custom config 파일

`cascade_rcnn_swinB_anchor.py`: 위 베이스 config 파일들을 통해 Swin Transformer를 backbone으로 활용하고 anchor scale을 추가한 custom config 파일


## 모델 특징:

`Swin Transformer`: 

![swin_img](https://kr.object.ncloudstorage.com/resume/boostcamp/swin.png)

Swin Transformer는 Transformer 구조를 컴퓨터 비전에 적용시키기 위한 백본 모델이다. Hierarchical Feature map 구조를 통해 Transformer를 다른 비전 모델처럼 계층적 구조로 표현이 가능하며, Shifted Window based Self-attention 구조를 통해 Feature map 해상도에 선형적인 복잡도를 가지는 Attention을 계산하고 이는 결국 계층적인 트랜스포머를 구현가능하게 한다. 

`Additional Anchor`: 비교적 크기가 작은 물체에 대한 정확도를 개선 시키기 위해 anchor scale을 기존의 8에서 4를 추가해 총 4와 8, 두 개의 anchor scale을 사용

`4 Convolution 1 FC`: 팀의 메인 모델과의 앙상블을 고려해 차이점을 주기 위해 bbox head로 4 convolution 1 fc head를 사용


## 모델 성능:

small object에 대한 mAP는 향상 되었지만 전체적인 mAP는 하락.

LB score: 0.657

