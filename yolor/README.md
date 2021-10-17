<h1 align="center">
<p>Yolor - You Only Learn One Representation: Unified Network for Multiple Tasks

</h1>

```bash
yolor
├──cfg
    ├── yolor_w6.cfg
    ├── yolor_p6.cfg
    ├── yolor_csp.cfg
    └── yolor_csp_x.cfg
└── models
    └── models.py
```
## 파일 설명:

`yolor_w6.cfg`: Model 관련 custom config 파일(Mixup)


## 모델 특징:

`Yolor`: 

![yolor_img](https://kr.object.ncloudstorage.com/resume/boostcamp/yolor.png)

yolor은 사람이 explicit knowledge와 implicit knowledge를 결합하여 정보를 처리하는 것을 모방해 두 지식을 함께 활용하는 모델을 제안합니다. 이 두 지식으로 다양한 task에 적합한 sub-representation을 지닌 모델을 학습합니다.


## 모델 성능:

Robust for Overfitting 

LB score: 0.611

### How to use
1. repo를 clone 합니다.
```
git clone https://github.com/WongKinYiu/yolor.git
```
2. Pretrained 된 이미지넷을 다운 받습니다. 
```
cd /yolor
bash scripts/get_pretrain.sh
```
3. cv_train.json을 yolo 형식대로 바꿔주기 위해 convert2Yolo를 yolor 디렉토리 안에서 clone해줍니다. 
```
git clone https://github.com/ssaru/convert2Yolo.git
cd convert2Yolo
```
4. coco.names를 저희 껄로 바꿔주고 json을 convert2Yolo 내의 example.py로 yolo 형식으로 바꿔줍니다.
- train : cv_train1.json
```
python3 example.py --datasets COCO --img_path ../../dataset/train --label ../../dataset/cv_train1.json --convert_output_path ./ --img_type ".jpg" --manifest_path ./ --cls_list_file coco.names
``` 
- manifest.txt를 train.txt로 바꿔주고, 
- val : cv_val1.json
```
python3 example.py --datasets COCO --img_path ../../dataset/train --label ../../dataset/cv_val1.json --convert_output_path ./ --img_type ".jpg" --manifest_path ./ --cls_list_file coco.names
``` 
- manifest.txt를 val.txt로 바꿔주고, 
- test : test.json
```
python3 example.py --datasets COCO --img_path ../../dataset/test --label ../../dataset/test.json --convert_output_path ./ --img_type ".jpg" --manifest_path ./ --cls_list_file coco.names
``` 
- mainfest.txt를 test.txt로 바꿔줍니다. 
5. train.txt, val.txt, test.txt를 상위 디렉토리인 yolor로 옮겨줍니다. 
6. yolor/data 로 들어가서 coco.names를 저희 클래스 목록으로 바꿔주고, coco.yaml을 아래와 같이 바꿔줍니다. 
- coco.yaml
```
train : ./train.txt
val : ./val.txt
test : ./test.txt

nc: 10
names : ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
```
7. train.py로 학습을 시작합니다! (Epoch은 많을 수록 좋음, 300이 기본이며 1일 정도 소요, 1000까지 돌려도 좋을 듯함. )
- train.py
```
python3 train.py --batch-size 8 --img 1280 --data coco.yaml --cfg cfg/yolor_w6.cfg --weights ./yolor_w6.pt --device 0 --name yolor_fold1 --hyp hyp.finetune.1280.yaml --epochs 900
```
8. test.py로 추론합니다. 
- test.py
```
python3 test.py --data data/coco.yaml --img 1280 --batch 32 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_w6.cfg --weights {weight 경로} --name yolor_fold1 --task test --verbose --save-conf --save-txt
```
9. 만들어진 txt를 제출양식으로 바꾸기 위해 아래 script를 사용합니다. 
- submit.py

```python
from glob import glob
import pandas as pd

images = sorted(glob("./runs/test/exp30/labels/*.txt"))
print(images[0])

image_id = []
PredictionString = []

df = pd.DataFrame()
for i in range(0, len(images)):
    sub_prediction = []
    if i % 100==0:
        print(images[i])
    image_id.append("test/"+images[i].split("/")[-1].replace("txt", "jpg"))
    with open(images[i]) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split("\n")[0].split(" ")
            class_id = l[0]
            x, y, w, h = l[1],l[2],l[3],l[4]
            conf = l[5]
            sub_prediction.append(class_id)
            sub_prediction.append(conf)
            sub_prediction.append(str(float(x) * 1024 - float(w) * 512))
            sub_prediction.append(str(float(y) * 1024 - float(h) * 512))
            sub_prediction.append(str(float(x) * 1024 + float(w) * 512))
            sub_prediction.append(str(float(y) * 1024 + float(h) * 512))
    # print(sub_image_id)
    if i % 1000==0:
        print(" ".join(sub_prediction))
    PredictionString.append(" ".join(sub_prediction))

df["PredictionString"] = PredictionString
df["image_id"] = image_id
print(df.head())
df.to_csv("submission.csv", index=None)
    
```