# object-detection-level2-cv-03

# 1. Introduction  
<br/>
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo.png" />
</p>
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo2.png"/>
</p>

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 5개월간의 교육과정입니다. 전체 과정은 이론과정(U-stage, 5주)와 실무기반 프로젝트(P-stage, 15주)로 구성되어 있으며, 두 번째 대회인 `Object detection`과제에 대한 **Level2 - 03조** 의 문제해결방법을 기록합니다.
  
<br/>

## 🧙‍♀️ Dobbyision - 도비도비전잘한다  
”도비도 비전을 잘합니다”  
### 🔅 Members  

김지수|박승찬|박준수|배지연|이승현|임문경|장석우
:-:|:-:|:-:|:-:|:-:|:-:|:-:
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]|![image6][image6]|![image7][image7]
[Github](https://github.com/memesoo99)|[Github](https://github.com/ark10806)|[Github](https://github.com/JJONSOO)|[Github](https://github.com/jiiyeon)|[Github](https://github.com/lsh3163)|[Github](https://github.com/larcane97)|[Github](https://github.com/sw-jang)


### 🔅 Contribution  
- `김지수` &nbsp; Data Synthesis • Model Searching • Model Experiment  
- `박승찬` &nbsp; Custom Dataset • Pseudo Labeling • Model Searching • Model Experiment • Ensemble   
- `박준수` &nbsp; Data Synthesis • Model Searching • Model Experiment • Ensemble  
- `배지연` &nbsp; Model Evaluation • Document Recording  
- `이승현` &nbsp; EDA • Modeling • Model Experiment • Ensemble
- `임문경` &nbsp; EDA • Data Augmentation • Model Searching • Model Experiment  
- `장석우` &nbsp; EDA • Modeling • Model Experiment • Ensemble 

[image1]: https://kr.object.ncloudstorage.com/resume/boostcamp/00.png
[image2]: https://kr.object.ncloudstorage.com/resume/boostcamp/01.png
[image3]: https://kr.object.ncloudstorage.com/resume/boostcamp/02.png
[image4]: https://kr.object.ncloudstorage.com/resume/boostcamp/03.png
[image5]: https://kr.object.ncloudstorage.com/resume/boostcamp/04.png
[image6]: https://kr.object.ncloudstorage.com/resume/boostcamp/05.png
[image7]: https://kr.object.ncloudstorage.com/resume/boostcamp/06.png


<br/>

# 2. Project Outline  

![competition_title](https://kr.object.ncloudstorage.com/resume/boostcamp/competition.png)

- Task : Object detection
- Date : 2021.09.27 - 2021.10.15 (3 weeks)
- Description : 쓰레기 사진을 입력받아서 `일반 쓰레기, 플라스틱, 종이, 유리 등`를 추측하여 `10개의 class`로 분류하고 박스의 영역을 구합니다.   
- Image Resolution : (1024 x 1024)
- Train : 4,833
- Test : 4,871

![objecteda](https://kr.object.ncloudstorage.com/resume/boostcamp/objecteda.png)


### 🏆 Final Score  
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/leaderboard.png">
</p>

<br/>

# 3. Solution
![process][process]

### KEY POINT
- 클래스의 불균형 문제가 모델의 성능에 큰 영향을 미치지 않습니다
- 오히려 객체의 수가 가장 많은 Paper 클래스에 대한 AP가 낮게 평가됩니다. 
- Small object 문제를 해결하는게 핵심입니다. 

&nbsp; &nbsp; → 주요 논점을 해결하는 방법론을 제시하고 실험결과를 공유하며 토론을 반복했습니다   

[process]: https://kr.object.ncloudstorage.com/resume/boostcamp/pipeline.png
<br/>

### Checklist
More Detail : https://kr.object.ncloudstorage.com/resume/boostcamp/CV3%EC%A1%B0%20%EB%B0%9C%ED%91%9C.pdf
- [x] Test Time Augmentation
- [x] Ensemble(Universenet, Swin, YoloR, Yolov5 등)
- [x] Augmentation(background patches, cutmix)
- [x] Multi-scale learning
- [x] Oversampling
- [x] Custom anchor ratio
- [x] Pseudo labeling
- [x] Collage
- [x] Stratified Kfold
- [x] Transfer learning(2 stage training)
- [ ] Ray
- [ ] Semi-supervised learning

### Evaluation

| Method| mAP| K-fold|
| --- | --- | --- |
|cascade RCNN + swin| 0.677| 0.704|
|CBNet|0.584|-|
|UniverseNet|0.594|0.604|
|YoloR|0.611|- |
|Yolov5|0.572|-|
|VFNet|0.562|-|
|HTC|0.647|-|

# 4. How to Use


```
.
├──/dataset
|   ├── train.json
|   ├── test.json
|   ├── /train
|   ├── /test
├──/object-detection-level2-cv-03
│   ├── utils
│   ├── model1
│         ├── config.py
│         └── readme.md
│   ├── model2
│         ├── config.py
│         └── readme.md
```

- `model`안에는 각각 **config.py** •  **readme.md**가 들어있습니다  
- `utils` 에는 앙상블, stratified k-fold, cutmix, 콜라주 기법 등 자체 구현한 util 모듈들이 포함되어 있습니다.
- 사용자는 전체 코드를 내려받은 후 설명서에 따라 옵션을 지정하여 개별 라이브러리의 모델을 활용할 수 있습니다
- 각 라이브러리의 구성요소는 `readme.md`에서 확인할 수 있습니다  
