import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, gauss
from collections import Counter
import copy

''' Todo
1. num_box 작은 순으로 Aug
2. class imbalance 고려
3. 
'''
imsize = 1024
margin = 16
n_class = 10
# dstroot = './aug'
dstroot = '/opt/ml/detection/datasetBGAug/'
bboxroot = '/opt/ml/detection/datasetBGAug/bbox'

if not os.path.isdir(dstroot):
    os.makedirs(bboxroot)
    os.makedirs(os.path.join(dstroot, 'train'))
    os.makedirs(os.path.join(dstroot, 'test'))

base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'train_v1.json')
train_json = json.load(open(train_path, 'r'))

images = train_json['images']
new_images = []
annots = train_json['annotations']
init_annot_len = len(annots) - 1
cls_annots = [[] for _ in range(n_class)]
for ann in annots:
    cls_annots[ann['category_id']].append(ann)

bg_patient = 7

def get_cls_panel():
    ann_cnt = Counter()
    for ann in annots:
        ann_cnt.update([ann['category_id']])
    
    ann_cnt = sorted(ann_cnt.items(), key=lambda x: x[0])
    ann_cnt = [x[1] for x in ann_cnt]
    ratio = np.array(ann_cnt)
    ratio = np.log(ratio + 1.1)
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

def get_coord(src_img_id, bbox, is_bg=False, patience=int(1024*1024*1/100)):
    _,_,w,h = bbox
    obj_range = get_range(src_img_id)
    if is_bg:
        p_margin = 0
        patience = int(1024*1024 * bg_patient / 100)
    elif imsize-w < margin+30 or imsize-h < margin+30:
        return False
    else:
        p_margin = margin

    for _ in range(patience):
        x, y = randint(p_margin, int(imsize-w) - p_margin), randint(p_margin, int(imsize-h)-p_margin)
        is_passed = True
        for _x, x_, _y, y_ in obj_range:
            if not ((x+w) < _x or x > x_) or ((y+h) < _y or y > y_):
                is_passed = False
                break
        if is_passed and not is_bg:
            annots_dic[src_img_id].append([x,y,int(w),int(h)])
        if is_passed:
            return [x, y, int(w), int(h)]
    return False

# def get_mix_coord(src_img_id, bbox, is_bg=False, patience=int(1024*1024*1/100)):
#     _,_,w,h = bbox
#     obj_range = get_range(src_img_id)
#     if is_bg:
#         p_margin = 0
#     elif imsize-w < margin+30 or imsize-h < margin+30:
#         return False
#     else:
#         p_margin = margin

#     for _ in range(patience):
#         x, y = randint(p_margin, int(imsize-w) - p_margin), randint(p_margin, int(imsize-h)-p_margin)
#         is_passed = True
#         for _x, x_, _y, y_ in obj_range:
#             if not ((x+w) < _x or x > x_) or ((y+h) < _y or y > y_):
#                 is_passed = False
#                 break
#         if is_passed and not is_bg:
#             annots_dic[src_img_id].append([x,y,int(w),int(h)])
#         if is_passed:
#             return [x, y, int(w), int(h)]
#     return False

img_dic = {}
img_ids = []
for imgj in images:
    img_dic[imgj['id']] = imgj['file_name']
    img_ids.append(imgj['id'])

annots_dic = {}
for ann in annots:
    if not ann['image_id'] in annots_dic.keys():
        annots_dic[ann['image_id']] = [ann['bbox']]
    else:
        annots_dic[ann['image_id']].append(ann['bbox'])
def get_range(patch_img_id):
    obj_range = []
    for bbox in annots_dic[patch_img_id]:
        x,y,w,h = bbox
        obj_range.append((x-margin, x+w+margin, y-margin, y+h+margin))
    return obj_range
def rand_bg_patch(patience=int(imsize*imsize*10/100)):
    w,h = int(gauss(200, 100)), int(gauss(200, 100))
    # w,h = int(gauss(257.50, 204.58)), int(gauss(252.24, 200.14))
    if w < 0 or h < 0:
        w, h = 200, 200
    patch_img_id = img_ids[randint(0, len(images)-1)]
    obj_range = get_range(patch_img_id)

    for _ in range(patience):
        x, y = randint(0, int(imsize-w)), randint(0, int(imsize-h))
        is_passed = True
        for _x, x_, _y, y_ in obj_range:
            if not ((x+w) < _x or x > x_) or ((y+h) < _y or y > y_):
                is_passed = False
                break
        if is_passed:
            img = Image.open(os.path.join(base, img_dic[patch_img_id]))
            bg_patch = img.crop((x, y, x+w, y+h))
            return bg_patch, [x,y,w,h]
    return False, False

def get_patch():
    patch_cls = cls_panel[ randint(0, len(cls_panel)-1) ]
    patch_annot = cls_annots[patch_cls][randint(0, len(cls_annots[patch_cls])-1)]
    img_id = patch_annot['image_id']

    img = Image.open(os.path.join(base, img_dic[img_id]))
    x,y,w,h = patch_annot['bbox']
    patch = img.crop((x,y,int(x+w),int(y+h)))

    return patch, patch_annot, patch_cls




cls_panel = get_cls_panel()
global ann_id
ann_id = 23144
def cutmix2bg(src_img_id, dst_img_id, fname, n_patch_max, n_bg_patch_max, n_mix_patch_max):
    global ann_id
    is_patched = 0
    src = Image.open(os.path.join(base, img_dic[src_img_id]))
    # bg = getbg(src_img_id)

    for ann in annots:
        base_annots = []
        if ann['image_id'] == src_img_id:
            ann_id += 1
            dup_ann = copy.deepcopy(ann)
            dup_ann['image_id'] = dst_img_id
            dup_ann['id'] = ann_id
            base_annots.append(dup_ann)
        annots.extend(base_annots)

    for _ in range(randint(0, n_bg_patch_max)):
        bg_patch, bg_bbox = rand_bg_patch()
        if bg_bbox:
            coord = get_coord(src_img_id, bg_bbox, is_bg=True)
            if coord:
                src.paste(bg_patch, coord[:2])

    for _ in range(randint(0, n_patch_max)):
        patch, patch_annot, patch_cls = get_patch()
        coord = get_coord(src_img_id, patch_annot['bbox'])
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
            # bg[x:x+w, y:y+h] = 1

    # for _ in range(randint(0, n_mix_patch_max)):
    #     patch, patch_annot, patch_cls = get_patch()
    #     coord = get_coord(src_img_id, patch_annot['bbox'])
    #     if coord :
    #         is_patched += 1
    #         ann_id += 1
    #         src.paste(patch, coord[:2])
    #         x,y,w,h = coord
    #         annots.append({
    #             'image_id': dst_img_id,
    #             'category_id': patch_cls, 
    #             'area': coord[2]*coord[3], 
    #             'bbox': [float(x), float(y), float(w), float(h)], 
    #             'iscrowd': 0, 
    #             'id': ann_id})

    if is_patched:
        src.save(os.path.join(dstroot, 'train', f'{fname}.jpg'))
        patch, patch_annot, patch_cls = get_patch()
        
    return is_patched



def main():
    n_aug = 10000
    n_patch_max = 6
    n_bg_patch_max = 6
    n_mix_patch_max = 2
    dst_img_id = 4883
    fname = 5000
    for _ in tqdm(range(n_aug)):
        src_img_id = img_ids[randint(0, len(images)-1)]
        is_patched = cutmix2bg(src_img_id, dst_img_id, fname, n_patch_max, n_bg_patch_max, n_mix_patch_max)
        if is_patched:
            new_images.append(
                {'width': 1024, 'height': 1024,
                    'file_name': f'train/{fname}.jpg',
                    'license': 0, 'flickr_url': None, 'coco_url': None, 'date_captured': '2021-01-10 16:30:39', 
                    'id': dst_img_id})
        fname += 1
        dst_img_id += 1
    
    train_json['images'] = images + new_images
    train_json['annotations'] = annots
    with open(os.path.join(dstroot, 'train_aug.json'), 'w') as f:
        json.dump(train_json, f)



font = ImageFont.truetype('/opt/ml/detection/object-detection-level2-cv-03/1Phase/EDA/ubuntu.regular.ttf', 15)
color_group = {'General trash': 'orange', 'Paper': 'white', 'Paper pack': 'ivory', 'Metal': 'dimgrey', 'Glass': 'dodgerBlue', 'Plastic': 'darkolivegreen', 'Styrofoam': 'khaki', 'Plastic bag': 'Teal', 'Battery': 'lime', 'Clothing': 'fuchsia'}
cates = []
for v in train_json['categories']:
    cates.append(v['name'])

def drawbox(draw, cate, pt1, pt2):
    draw.rectangle((pt1, pt2), outline=color_group[cate], width=3)
    draw.text(pt1, cate, (255,255,255), font=font, backrec=True)

def draw():
    images = new_images
    for image in tqdm(images):
        img = Image.open(os.path.join(dstroot, image['file_name']))
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
    print(len(new_images))