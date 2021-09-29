import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from random import randint
from collections import Counter
import copy

##
# Todo
'''
1. num_box 작은 순으로 Aug
2. class imbalance 고려
3. 
'''
imsize = 1024
margin = 30
n_class = 10
base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'train_v1.json')

train_json = json.load(open(train_path, 'r'))

images = train_json['images']
annots = train_json['annotations']
init_annot_len = len(annots)
cls_annots = [[] for _ in range(n_class)]
for ann in annots:
    cls_annots[ann['category_id']].append(ann)


def get_cls_panel():
    ann_cnt = Counter()
    for ann in annots:
        ann_cnt.update([ann['category_id']])
    
    ann_cnt = sorted(ann_cnt.items(), key=lambda x: x[0])
    ann_cnt = [x[1] for x in ann_cnt]
    ratio = np.array(ann_cnt)
    ratio = np.log(ratio + 1.9)
    ratio = ratio / np.sum(ratio)
    ratio = 1 - (ratio / np.max(ratio))

    cls_panel = []
    for idx, rat in enumerate(ratio):
        for _ in range(int(rat*5000)):
            cls_panel.append(idx)
    
    return cls_panel

def getbg(img_id):
    # img = Image.open(src)
    bg = np.zeros(shape=(imsize,imsize))
    bbox = []
    for ann in annots:
        if ann['image_id'] == img_id:
            x,y,w,h = ann['bbox']
            bbox.append([int(x-margin), int(y-margin), int(x+w+margin), int(y+h+margin)])

    for x,y,w,h in bbox:
        bg[x:x+w, y:y+h] = 1
    return bg

def get_coord(bg, bbox, patience=int(1024*1024*1/100)):
    _,_,w,h = bbox
    for _ in range(patience):
        x, y = randint(0, int(imsize-w)), randint(0, int(imsize-h))
        if np.sum(bg[x:int(x+w), y:int(y+h)]) == 0:
            return [x,y,int(w),int(h)]
    return False

def get_patch():
    patch_cls = cls_panel[ randint(0, len(cls_panel)-1) ]
    patch_annot = cls_annots[patch_cls][randint(0, len(cls_annots[patch_cls])-1)]
    img_id = patch_annot['image_id']
    for imgj in images:
        if imgj['id'] == img_id:
            img_path = imgj['file_name']
            break
    
    img = Image.open(os.path.join(base, img_path))
    x,y,w,h = patch_annot['bbox']
    patch = img.crop((x,y,int(x+w),int(y+h)))
    # patch = patch.resize((int(w), int(h)))

    return patch, patch_annot, patch_cls

img_dic = {}
img_ids = []
for imgj in images:
    img_dic[imgj['id']] = imgj['file_name']
    img_ids.append(imgj['id'])


cls_panel = get_cls_panel()
def cutmix2bg(src_img_id, dst_img_id, ann_id, n_patch):
    is_patched = 0
    src = Image.open(os.path.join(base, img_dic[src_img_id]))
    bg = getbg(src_img_id)

    for _ in range(randint(0, n_patch)):
        patch, patch_annot, patch_cls = get_patch()
        coord = get_coord(bg, patch_annot['bbox'])
        if coord :
            is_patched += 1
            ann_id += 1
            src.paste(patch, coord[:2])
            annots.append({
                'image_id': dst_img_id,
                'category_id': patch_cls, 
                'area': coord[2]*coord[3], 
                'bbox': coord, 'iscrowd': 0, 
                'id': ann_id})
            x,y,w,h = coord
            bg[x:x+w, y:y+h] = 1
    if is_patched:
        ##########DST로 바꿔야 함
        src.save(f'./aug/{src_img_id}_{is_patched}.jpg')
    return ann_id

def main():
    n_aug = 100
    n_patch_max = 10
    ## json 저장해야함
    dst_img_id = len(images)
    ann_id = len(annots)
    for _ in tqdm(range(n_aug)):
        src_img_id = img_ids[randint(0, len(images)-1)]
        n_patch = randint(0, n_patch_max)
        ann_id = cutmix2bg(src_img_id, dst_img_id, ann_id, n_patch)
        dst_img_id += 1

if __name__ == '__main__':
    main()