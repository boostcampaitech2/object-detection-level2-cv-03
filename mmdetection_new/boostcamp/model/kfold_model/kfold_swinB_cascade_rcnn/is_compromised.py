'''
- data_root와 fold_num을 알맞게 기입해 주세요.
- isCompromised가 0% 이면 train, val이 섞이지 않은 것입니다.
- 0이 아닌 다른 숫자가 나오면 train, val이 섞인 상태입니다.
'''

import json
import os

data_root = '/opt/ml/detection/mmdetection/boostcamp/kfold_model/kfold_swinB_cascade_rcnn/kfold_dataset/'
fold_num = 2

def is_compromised(train_path, val_path):
    train = json.load(open(train_path, 'r'))
    val = json.load(open(val_path, 'r'))

    train_imgs = train['images']
    val_imgs = val['images']

    train_set = set()
    val_set = set()

    for img in train_imgs:
        train_set.update([ img['file_name'] ])
    for img in val_imgs:
        val_set.update([ img['file_name'] ])
    compromised = train_set.intersection(val_set)

    print(f'\n\nstarts k-fold with {data_root}cv_train{fold_num}.json')
    print(f'\nisCompromised: {len(compromised)}%\n\n')
    assert len(compromised) == 0

train_path = os.path.join(data_root, f'cv_train{fold_num}.json')
val_path = os.path.join(data_root, f'cv_val{fold_num}.json')
is_compromised(train_path, val_path)