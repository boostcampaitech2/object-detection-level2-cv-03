# Utils

## /cutmix
- cutmmix module이 구현되어 있습니다.
- 사용법은 /cutmix 내부의 readme.md를 참고


## /dataset_eda
- box_ratio_box_area_eda.ipynb : dataset의 box ratio와 area의 비율을 분석한 eda파일
- cutmix_aug.ipynb : /cutmix에서 구현된 cutmix module의 결과물을 시각화한 파일
- fiftyone_eda.ipynb : 시각화 툴 중 하나인 fiftyone의 기본적인 기능을 재활용 데이터셋에 적용한 파일


## /visualization
- mm_inference.ipynb : 모델 inference 후 fiftyone을 이용한 시각화까지 포함된 파일
- submission_visualization.ipynb : 주어진 submission.csv를 기반으로 fiftyone을 통한 시각화를 해주는 파일


## /stratified_fold
- stratified_kfold.py : 기존의 train.json을 stratified kfold로 변환시켜주는 파일


## /pseudo_kfold
- pseudo_label.py : submission.csv를 coco format의 json파일로 변환시켜주는 파일
- pseudo_stratified_kfold.py : 위에서 변환된 pseudo label json을 stratified kfold로 변환시켜주는 파일


## /notebook
- .ipynb기반의 여러 파일들이 들어있는 폴더
- inference, visualization, ensemble 등의 파일 존재

## /data_systhesis
- bbox를 category별로 cutout해서 새로운 합성 data를 만드는 코드
- 흰 캔버스에 합성하는 버전과 배경패치에 이어 붙이는 버전 두가지가 있으나 배경패치 data는 해당 레포에서 제공하지 않는다.
