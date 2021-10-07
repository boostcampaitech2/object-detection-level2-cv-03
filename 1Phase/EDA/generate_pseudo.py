import os
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
from random import randint, shuffle

bgroot = '/opt/ml/detection/object-detection-level2-cv-03/1Phase/EDA/bgs/BG'
bgs = glob(f'{bgroot}/*')
lenbgs = len(bgs)

patchroot = '/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/work_dirs/swinB_Rmc5121024_4c1f_fullAugs+_defualtDS_LB661/ensembles/cutsrcB_B+pse_Univ_LB674'
model_name = 'B_B+pse_Univ_LB674'
dstroot = os.path.join('/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/work_dirs/swinB_Rmc5121024_4c1f_fullAugs+_defualtDS_LB661/ensembles/pseudo/', model_name)
if not os.path.isdir(dstroot):
    os.makedirs(dstroot)
    os.makedirs(os.path.join(dstroot, 'train'))
    os.makedirs(os.path.join(dstroot, 'test'))

base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'train_v1.json')
train_json = json.load(open(train_path, 'r'))

# images = train_json['images']
# annots = train_json['annotations']

images = []
annots = []

class Patch():
    def __init__(self, patchroot):
        self.patch_pathes = glob(os.path.join(patchroot, '*'))
        self.total_len = len(self.patch_pathes)
        self.patient = 100
        self.failed = 0
        self.prevlen = 0
        print(self.total_len)

    def get_patches(self, n):
        """ Get Image whose confidence is higher than confidence_thres (=0.98)

        Args:
            n ([type]): number of patches to paste one BG image.

        Returns:
            patches [str]: patch path
            isFull  [bool]: annots if the iteration is full or not
        """
        lenRests = len(self.patch_pathes)

        if self.failed == self.patient:
            return False, True
        if self.prevlen == len(self.patch_pathes):
            self.failed += 1
        
        self.prevlen = lenRests


        print(f'[{self.total_len - lenRests: <5}/{self.total_len: <5}]: {(self.total_len - lenRests)/self.total_len * 100: .2f}%', end='')
        if n < lenRests:
            patches = []
            for _ in range(n):
                patches.append(self.patch_pathes.pop(randint(0, len(self.patch_pathes)-1)))
            return patches, False

        else:
            patches = self.patch_pathes
            return patches, True

    def update_iter(self, path):
        self.patch_pathes.insert(-1, path)

patchcls = Patch(patchroot)

n2paste = randint(1, 5)
patch_paths, isFull = patchcls.get_patches(n2paste)

ann_id = 0
def paste_patch(bg: Image, path: Image, reserved, iteration):
    global ann_id
    patch = Image.open(path)
    w,h = patch.size

    anchors = []
    for i in range(int(w/10), 1024-w, 8):
        for j in range(int(h/10), 1024-h, 8):
            anchors.append((i, j))
    shuffle(anchors)

    for x, y in anchors:
        is_overlapped = False
        for x0, y0, w0, h0 in reserved:
            if not ((x+w) < x0 or x > x0+w0) or ((y+h) < y0 or y > y0+h0):
                is_overlapped = True
    
        if not is_overlapped:
            bg.paste(patch, (x, y))
            annots.append({
                'image_id': iteration,
                'category_id': int(path.split('/')[-1].split('.')[-2].split('_')[-1]),
                'area': w*h, 
                'bbox': [float(x), float(y), float(w), float(h)], 
                'iscrowd': 0, 
                'id': ann_id
            })
            ann_id += 1
            reserved.append([x, y, w, h])
            return bg, 1, reserved
    
    patchcls.update_iter(path)
    return bg, 0, reserved


iteration = 0
n2paste = 10
patch_paths, isFull = patchcls.get_patches(n2paste)
while not isFull:
    bg = Image.open(bgs[randint(0, lenbgs-1)])
    n_patched = 0
    reserved = []
    for path in patch_paths:
        bg, nn, reserved = paste_patch(bg, path, reserved, iteration)
        n_patched += nn
    print(f'\t{n_patched} / {n2paste}', end='')
    if n_patched != 0:
        bg.save(os.path.join(dstroot, 'pseudo_train', f'pse_{iteration}.jpg'))
        images.append({'width': 1024, 'height': 1024,
                    'file_name': f'train/pse_{iteration}.jpg',
                    'license': 0, 'flickr_url': None, 'coco_url': None, 'date_captured': '2021-01-10 16:30:39', 
                    'id': iteration})
        print('\tsaved!', end='')
        iteration += 1
    print(f'iter: {iteration}\t | {len(images)}')
    patch_paths, isFull = patchcls.get_patches(n2paste)
    if not patch_paths:
        print('break')
        break


train_json['images'] = images
train_json['annotations'] = annots
with open(os.path.join(dstroot, f'pseudo_{model_name}2.json'), 'w') as f:
    json.dump(train_json, f)