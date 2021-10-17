<h1 align="center">
<p>Universenet, CBNetV2
</h1>

```bash
Universenet
 ├── _base_
 │    ├── custom_dataset.py
 │    ├── custom_runtime.py
 │    └─ custom_schedule.py
 └── model
      ├── univ101.py
      └─── cbnet.py
```
## 파일 설명:

`custom_dataset.py`: Dataset 관련 custom config 파일

`custom_schedule.py`: Cosine Annealing을 적용한 schedule 관련 custom config 파일

`custom_runtime.py`: log와 evaluation 관련 custom config 파일

`univ101.py`: backbone resenet101을 사용한 universenet config 파일

`cbnet.py`: backbone CBRes2Net을 사용한 CBNetV2 config 파일


## 모델 특징:

`CBNetV2`: 

![swin_img](https://kr.object.ncloudstorage.com/resume/boostcamp/cbnetv2.png)

CBNetV2는 backbone 에서 특징을 뽑는것이 object detection 에서 매우 중요한 전략이라고 생각하여 선학습된 여러가지 백본을 조합하고 크기가 다른 특징맵을 다양하게 통합해서 검출 성능을 높인 모델이다.

`Universenet`: ATSS (Adaptive Training Sample Selectionand ) Anchor-free, Anchor-based 의 차이는 positive , negative samples의 정의에 따라 성능이 갈린다. 이에따라 object 특성에 따라 positive , negative sample을 선택하는 알고리즘을 사용하였고 SEPC without iBN 사용하여 성능을 올렸다.



## 모델 성능:

Universenet101 LB score: 0.594 , k-fold LB scre : 0.604

CBNetV2 LB score : 0.562
