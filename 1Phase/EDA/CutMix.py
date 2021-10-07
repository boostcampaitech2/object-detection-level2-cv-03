import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, gauss
from collections import Counter
import copy
import time

''' Todo
1. 누더기 배경
2. cutmix
3. 3
'''
imsize = 1024
margin = 16
n_class = 10
# dstroot = './aug'
dstroot = '/opt/ml/detection/Augmented/dataset'
bboxroot = os.path.join(dstroot, 'bbox')

if not os.path.isdir(dstroot) or not os.path.isdir(bboxroot):
    os.makedirs(bboxroot)
    os.makedirs(os.path.join(dstroot, 'train'))
    os.makedirs(os.path.join(dstroot, 'test'))

base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'train_v1.json')
train_json = json.load(open(train_path, 'r'))

images = train_json['images']
new_images = []
annots = train_json['annotations']
init_img_len = len(images) -1
init_annot_len = len(annots) - 1
cls_annots = [[] for _ in range(n_class)]
for ann in annots:
    cls_annots[ann['category_id']].append(ann)


annots_dic = {}
for ann in annots:
    if not ann['image_id'] in annots_dic.keys():
        annots_dic[ann['image_id']] = [ann['bbox']]
    else:
        annots_dic[ann['image_id']].append(ann['bbox'])


ann_dict = {}
for ann in annots:
    ann_dict[ann['image_id']] = ann

bg_patient = 3

def get_background():
    src = images[randint(0, init_img_len)]
    



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

def get_coord(src_img_id, bbox, is_bg=False, patience=int(1024*1024/100)):
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
def rand_bg_patch(patience=int(imsize*imsize/100)):
    w,h = int(gauss(180, 100)), int(gauss(180, 100))
    # w,h = int(gauss(257.50, 204.58)), int(gauss(252.24, 200.14))
    if w < 0 or h < 0:
        w, h = 180, 180
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


def calc_IoU(box1, box2):
    x0,y0,x1,y1 = box1
    a0,b0,a1,b1 = box2

    if (a0 > x0 and b0 > y0) and (a1 < x1 and b1 < y1):
        AnB = (a1-a0) * (b1-b0)
        AuB = (x1-x0) * (y1-y0)
    elif not((x0< a0 < x1) or (x0 < a1 < x1) or (y0 < a0 <y1) or (y0 < a1 <y1)):
        return 1
    else:
        if b0 < y0:
            if a0 < x0:
                AnB = (x0-a1) * (y0-b1)
            else:
                AnB = (x1-a0) * (y0-b1)
        else:
            if a0 < x0:
                AnB = (x0-a1) * (y1-b0)
            else:
                AnB = (x1-a0) * (y1-b0)
        AnB = abs(AnB)
        AuB = (x1-x0) * (y1-y0)
    if AuB <= 0 :
        return 0
    else:
        return AnB / AuB

def get_mix_patch(src_img_id, patient=int(imsize*imsize*10/100)):
    patch, patch_annot, patch_cls = get_patch()

    x0, y0 = 1024, 1024
    x1, y1 = 0, 0
    for x,y,w,h in annots_dic[src_img_id]:
        if (x0 > x): x0 = x
        if (y0 > y): y0 = y
        if (x1 < x+w): x1 = x + w
        if (y1 < y+h): y1 = y + h
        
    _,_,w,h = patch_annot['bbox']
    a0,b0,a1,b1 = 0,0,0,0
    ov_cnt = 0
    iou = 0
    patient_cnt = 0
    while not ((a0>0 and a1<imsize and b0>0 and b1<imsize) and (0.1 < iou < 0.5) and (ov_cnt == 1) or (patient_cnt >= patient)):
        if x1+w >= imsize:
            x1 = imsize-w
        if y1+h >= imsize:
            y1 = imsize-h
        if x1 < x0 or y1 < y0:
            return False, False, False
        a0, b0 = randint(int(x0 - w), int(x1)), randint(int(y0 - h), int(y1))
        a1, b1 = int(a0 + w), int(b0 + h)

        ious = []
        for x,y,w_t,h_t in annots_dic[src_img_id]:
            ious.append(calc_IoU([x,y,x+w_t,y+h_t], [a0,b0,a1,b1]))
        ov_cnt = 0
        for iou in ious:
            if (iou != 0): ov_cnt+=1
        patient_cnt += 1
    
    if patient_cnt == patient:
        return False, False, False
    else:
        return patch, [a0, b0, w, h], patch_cls


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

    for _ in range(randint(0, n_mix_patch_max)):
        patch, coord, patch_cls = get_mix_patch(src_img_id)
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
            annots_dic[src_img_id].append(coord)

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

    

    if is_patched:
        src.save(os.path.join(dstroot, 'train', f'{fname}.jpg'))
        patch, patch_annot, patch_cls = get_patch()
        
    return is_patched



def main():
    n_aug = 10000
    n_patch_max = 4
    n_bg_patch_max = 6
    n_mix_patch_max = 5
    dst_img_id = 4883
    fname = 5000
    num_new = 0
    for idx, _ in enumerate(tqdm(range(n_aug), desc=f'Aug: {len(new_images)}')):
        src_img_id = img_ids[randint(0, len(images)-1)]
        is_patched = cutmix2bg(src_img_id, dst_img_id, fname, n_patch_max, n_bg_patch_max, n_mix_patch_max)
        if is_patched:
            num_new += 1
            new_images.append(
                {'width': 1024, 'height': 1024,
                    'file_name': f'train/{fname}.jpg',
                    'license': 0, 'flickr_url': None, 'coco_url': None, 'date_captured': '2021-01-10 16:30:39', 
                    'id': dst_img_id})
        fname += 1
        dst_img_id += 1

        if idx % 100 == 0:
            print(f'[{num_new: <6} / {idx+1: <6}]: {num_new/(idx+1)*100: .2f}%')
    
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
    for image in tqdm(images, desc='drawing bbox'):
        img = Image.open(os.path.join(dstroot, image['file_name']))
        draw = ImageDraw.Draw(img)
        for ann in annots:
            if ann['image_id'] == image['id']:
                x,y,w,h = ann['bbox']
                pt1, pt2 = (x,y), (x+w, y+h)
                drawbox(draw, cates[ann['category_id']], pt1, pt2)
                img.save(os.path.join(bboxroot, image['file_name'].split('/')[-1]))

if __name__ == '__main__':
    startT = time.time()
    main()
    draw()
    print(len(new_images))
    print(f'{(time.time() - startT)/60: .2f}min')