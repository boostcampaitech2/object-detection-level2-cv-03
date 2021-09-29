import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
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
dstroot = './aug'
bboxroot = './bbox'

base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'train_v1.json')

train_json = json.load(open(train_path, 'r'))

images = train_json['images']
new_images = []
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
def cutmix2bg(src_img_id, dst_img_id, fname, ann_id, n_patch):
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
            x,y,w,h = coord
            annots.append({
                'image_id': dst_img_id,
                'category_id': patch_cls, 
                'area': coord[2]*coord[3], 
                'bbox': [float(x), float(y), float(w), float(h)], 
                'iscrowd': 0, 
                'id': ann_id})
            x,y,w,h = coord
            bg[x:x+w, y:y+h] = 1
    if is_patched:
        ##########DST로 바꿔야 함
        src.save(os.path.join(dstroot, f'{fname}.jpg'))
    print(is_patched)
    return ann_id, is_patched

def main():
    n_aug = 10
    n_patch_max = 10
    ## json 저장해야함
    dst_img_id = len(images)
    ann_id = len(annots)
    fname = 5000
    for _ in tqdm(range(n_aug)):
        src_img_id = img_ids[randint(0, len(images)-1)]
        n_patch = randint(0, n_patch_max)
        ann_id, is_patched = cutmix2bg(src_img_id, dst_img_id, fname, ann_id, n_patch)
        if is_patched:
            new_images.append(
                {'width': 1024, 'height': 1024,
                    'file_name': f'aug/{fname}.jpg',
                    # 'file_name': f'train/{fname}.jpg',
                    'license': 0, 'flickr_url': None, 'coco_url': None, 'date_captured': '2021-01-10 16:30:39', 
                    'id': dst_img_id})
        fname += 1
        dst_img_id += 1
    
    train_json['images'] = images + new_images
    train_json['annotations'] = annots
    with open(os.path.join(dstroot, 'train_aug.json'), 'w') as f:
        json.dump(train_json, f)



font = ImageFont.truetype('./ubuntu.regular.ttf', 15)
color_group = {'General trash': 'orange', 'Paper': 'white', 'Paper pack': 'ivory', 'Metal': 'dimgrey', 'Glass': 'dodgerBlue', 'Plastic': 'darkolivegreen', 'Styrofoam': 'khaki', 'Plastic bag': 'Teal', 'Battery': 'lime', 'Clothing': 'fuchsia'}
cates = []
for v in train_json['categories']:
    cates.append(v['name'])

def drawbox(draw, cate, pt1, pt2):
    draw.rectangle((pt1, pt2), outline=color_group[cate], width=3)
    backrec = (pt1[0]+90, pt1[1]+15)
    # draw.rectangle((pt1, backrec), outline=(0,0,0), fill=True)
    draw.text(pt1, cate, (255,255,255), font=font, backrec=True)

def draw():
    images = new_images
    for image in tqdm(images):
        img = Image.open(os.path.join('./', image['file_name']))
        draw = ImageDraw.Draw(img)
        for ann in annots:
            if ann['image_id'] == image['id']:
                x,y,w,h = ann['bbox']
                pt1, pt2 = (x,y), (x+w, y+h)
                drawbox(draw, cates[ann['category_id']], pt1, pt2)
                img.save(os.path.join(bboxroot, image['file_name'].split('/')[-1]))

if __name__ == '__main__':
    main()
    draw()