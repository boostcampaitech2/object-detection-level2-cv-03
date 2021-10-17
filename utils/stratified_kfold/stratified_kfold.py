#!pip install scipy
#!pip3 uninstall scikit-learn --yes
#!pip3 install scikit-learn==0.22
#!pip install iterative-stratification

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from operator import add
import numpy as np
import json
import os

'''
- X = 각 image

- Y = image에 존재하는 label

- image id에 대응하는 annotation label값을 모두 Y에 저장
    - len(X)=len(Y)= 데이터 개수
    
- skfold로 label을 균등하게 나눈 index값을 구함

- index = image_id이므로 해당 image_id를 가진 image와 annotations들을 모아서 값을 저장
'''

annos_path = '../../dataset/train.json'
with open(annos_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

X = coco['images']
Y = [ [0]*len(categories) for _ in range(len(images))]

for anno in annotations:
    image_id = anno['image_id']
    Y[image_id][anno['category_id']]+=1

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=1010)


imgid2annos = [[] for _ in range(len(coco['images']))]
for anno in annotations:
    imgid = anno['image_id']
    imgid2annos[imgid].append(anno)


root = '.'

for idx,(train_index, val_index) in enumerate(mskf.split(X, Y)):
    cv_train_path = os.path.join(root,f'cv_train{idx+1}.json')
    cv_val_path = os.path.join(root,f'cv_val{idx+1}.json')
    cv_train = dict()
    cv_val = dict()

    # train
    cv_train['info'] = coco['info']
    cv_train['licenses'] = coco['licenses']
    cv_train['categories'] = coco['categories']
    
    train_images=[]
    train_annos=[]
    for t_index in train_index:
        train_images.append(X[t_index])
        image_id = X[t_index]['id']
        train_annos +=imgid2annos[image_id]
    
    cv_train['images'] = train_images
    cv_train['annotations'] = train_annos
    
    with open(cv_train_path,'w') as f:
        json.dump(cv_train,f)
    
    # validation
    cv_val['info'] = coco['info']
    cv_val['licenses'] = coco['licenses']
    cv_val['categories'] = coco['categories']
    
    val_images=[]
    val_annos=[]
    for v_index in val_index:
        val_images.append(X[v_index])
        image_id = X[v_index]['id']
        val_annos +=imgid2annos[image_id]
    
    cv_val['images'] = val_images
    cv_val['annotations'] = val_annos
    
    with open(cv_val_path,'w') as f:
        json.dump(cv_val,f)

