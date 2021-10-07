import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, gauss
from collections import Counter
import copy

imsize = 1024
margin = 16
n_class = 10
dstroot = './bgs'

if not os.path.isdir(dstroot):
    os.makedirs(dstroot)

base = '/opt/ml/detection/dataset'
train_path = os.path.join(base, 'train_v1.json')
train_json = json.load(open(train_path, 'r'))

images = train_json['images']
annots = train_json['annotations']

annots_dic = {}
for ann in annots:
    if not ann['image_id'] in annots_dic.keys():
        annots_dic[ann['image_id']] = [ann['bbox']]
    else:
        annots_dic[ann['image_id']].append(ann['bbox'])


def get_bg_patch():
    cnt = 1
    for image in tqdm(images):
        img = Image.open(os.path.join(base, image['file_name']))

        x0, y0 = imsize, imsize
        x1, y1 = 0, 0
        for x,y,w,h in annots_dic[image['id']]:
            if (x0 > x): 
                x0 = x
            if (y0 > y): 
                y0 = y
            if (x1 < x+w):
                x1 = x + w
            if (y1 < y+h): 
                y1 = y + h
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # draw = ImageDraw.Draw(img)
        # draw.rectangle((x0,y0,x1,y1), outline=(0,0,0), width=4)
        # img.save(f'{dstroot}/{cnt}.jpg')
        # cnt+=1


        # (pt1, pt2)
        patches = [
            (0,0, x0,y0),
            (0,y0, x0,y1),
            (0,y1, x0,1024),

            (x0,0, x1,y0),
            (x0,y1, x1,1024),

            (x1,0, 1024,y0),
            (x1,y0, 1024,y1),
            (x1,y1, 1024,1024)
        ]

        margin = 30
        if x0 < margin:
            patches[0] = False
            patches[1] = False
            patches[2] = False
        if y0 < margin:
            patches[0] = False
            patches[3] = False
            patches[5] = False
        if 1024-x1 < margin:
            patches[5] = False
            patches[6] = False
            patches[7] = False
        if 1024-y1 < margin:
            patches[2] = False
            patches[4] = False
            patches[7] = False




        
        for pat in patches:
            if pat != False:
                cr = img.crop(pat)
                cr.save(f'{dstroot}/{cnt}.jpg')
                cnt += 1
        


get_bg_patch()