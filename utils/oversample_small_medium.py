import os
from PIL import Image
import json
from glob import glob
from random import randint, shuffle

bgroot = '/opt/ml/detection/Augs/BG'
bgs = glob(f'{bgroot}/*')
lenbgs = len(bgs)

s_patchroot = './s_patches'
m_patchroot = './m_patches'
dstroot = './sm_oversampled'
if not os.path.isdir(dstroot):
    os.makedirs(dstroot)
    os.makedirs(os.path.join(dstroot, 'train'))
    os.makedirs(os.path.join(dstroot, 'test'))

base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'cv_train1.json')
train_json = json.load(open(train_path, 'r'))

s_patches = glob(os.path.join(s_patchroot, '*'))
m_patches = glob(os.path.join(m_patchroot, '*'))
patchMix = []
patchMix.extend(s_patches)
patchMix.extend(s_patches)
patchMix.extend(m_patches)
shuffle(patchMix)

new_images = []
new_annots = []

ann_id = 0
def paste_patch(bg: Image, path: Image, reserved: list, iteration: int):
    """ This function generates train images
        that has small, medium sized patches on the 
        randomly selected background patched images.

    Args:
        bg (Image): background patched images.
        path (Image): small and medium sized patches
        reserved (list): helps to calcuate if objects are overlapped or not.
        iteration (int): 

    Returns:
        [type]: [description]
    """
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
            if not (((x+w) < x0 or x > x0+w0) and ((y+h) < y0 or y > y0+h0)):
                is_overlapped = True
    
        if not is_overlapped:
            bg.paste(patch, (x, y))
            new_annots.append({
                'image_id': iteration,
                'category_id': int(path.split('/')[-1].split('.')[-2].split('_')[-1]),
                'area': w*h, 
                'bbox': [float(x), float(y), float(w), float(h)], 
                'iscrowd': 0, 
                'id': ann_id
            })
            ann_id += 1
            reserved.append([x, y, w, h])
            return bg, 1, reserved, False
    
    return bg, 0, reserved, path



iteration = 0
Flag = True
while Flag:
    bg = Image.open(bgs[randint(0, lenbgs-1)])
    n_patched = 0
    reserved = []
    patch_paths = []
    rand_p = randint(7, 30)
    for _ in range(rand_p):
        if len(patchMix):
            patch_paths.append(patchMix.pop())
        else:
            Flag = False

    for path in patch_paths:
        bg, nn, reserved, returned = paste_patch(bg, path, reserved, iteration)
        n_patched += nn
        if returned:
            patchMix.append(returned)
    print(f'\t{n_patched: <3} / {rand_p: <3}', end='')
    if n_patched != 0:
        bg.save(os.path.join(dstroot, 'train', f'sm_{iteration}.jpg'))
        new_images.append({'width': 1024, 'height': 1024,
                    'file_name': f'train/sm_{iteration}.jpg',
                    'license': 0, 'flickr_url': None, 'coco_url': None, 'date_captured': '2021-01-10 16:30:39', 
                    'id': iteration})
        print('\tsaved!', end='')
        iteration += 1
    print(f'\t{len(patchMix)}')
    if not patch_paths:
        print('break')
        break


train_json['images'] = new_images
train_json['annotations'] = new_annots
with open(os.path.join(dstroot, f'pure_sm.json'), 'w') as f:
    json.dump(train_json, f)