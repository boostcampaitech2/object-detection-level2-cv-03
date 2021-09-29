import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from random import randint
from collections import Counter

##
# Todo
'''
1. num_box 작은 순으로 Aug
2. class imbalance 고려
3. 
'''
imsize = 1024
margin = 30
base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'train_v1.json')

train_json = json.load(open(train_path, 'r'))

images = train_json['images']
annots = train_json['annotations']

def get_cls_ratio():
    ann_cnt = Counter()
    for ann in annots:
        ann_cnt.update([ann['category_id']])
    
    ann_cnt = sorted(ann_cnt.items(), key=lambda x: x[0])
    ratio = np.array(ann_cnt.values())
    ratio = 1 - (ratio / np.max(ratio))
    return ratio


def getbg(img_id):
    # img = Image.open(src)
    bg = np.zeros(shape=(imsize,imsize))
    bbox = []
    for ann in annots:
        if ann['image_id'] == img_id:
            x,y,w,h = ann['bbox']
            bbox.append([int(x-margin), int(y-margin), int(w+margin), int(h+margin)])

    for b in bbox:
        x,y,w,h = b
        bg[x:x+w, y:y+h] = 1
    
    return bg

def get_vac(bg, bbox, patience=int(1024*1024*1/100)):
    x, y = -1, -1
    _,_,w,h = bbox
    for _ in range(patience):
        x, y = randint(0, imsize-w), randint(0, imsize-h)
        if np.sum(bg[x:x+w, y:y+h]) == 0:
            break
    
    if x!=-1 and y!=-1:
        return [x,y,w,h]
    return False

def cutmix2vacant(src, patch, n_append):
    for _ in n_append(n_append):
        

np.bitwise_or