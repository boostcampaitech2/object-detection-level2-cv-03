# object-detection-level2-cv-03

## /cutmix
- cutmmix module이 구현되어 있습니다.
- 사용법은 /cutmix 내부의 readme.md를 참고


## /dataset_eda
- box_ratio_box_area_eda.ipynb : dataset의 box ratio와 area의 비율을 분석한 eda파일
- cutmix_aug.ipynb : /cutmix에서 구현된 cutmix module의 결과물을 시각화한 파일
- fiftyone_eda.ipynb : 시각화 툴 중 하나인 fiftyone의 기본적인 기능을 재활용 데이터셋에 적용한 파일
- pseudo_labeling.ipynb : submission.csv를 coco format의 json파일로 변환시켜주는 파일
- pseudo_stratified_kfold.ipynb : 위에서 변환된 pseudo label json을 stratified kfold로 변환시켜주는 파일
- stratified_kfold.ipynb : 기존의 train.json을 stratified kfold로 변환시켜주는 파일

## /visualization
- mm_inference.ipynb : 모델 inference 후 fiftyone을 이용한 시각화까지 포함된 파일
- submission_visualization.ipynb : 주어진 submission.csv를 기반으로 fiftyone을 통한 시각화를 해주는 파일

## /model
- mmdetection기반의 모델 config가 들어있는 폴더

## /notebook
- .ipynb기반의 여러 파일들이 들어있는 폴더
- inference, visualization, ensemble 등의 파일 존재
