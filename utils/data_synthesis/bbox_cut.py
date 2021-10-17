import json
import pandas as pd
import cv2
import os
import math
import random
from PIL import Image
from PIL import ImageDraw
import xml.etree.ElementTree as etree
from itertools import combinations


dataset_base = 'dataset/train_collage'

with open('dataset/train.json') as json_file:
    train_json = json.load(json_file)
    
#class name    
classes = {"General_trash":0, "Paper":1, "Paper_pack":2, "Metal":3, "Glass":4, 
           "Plastic":5, "Styrofoam":6, "Plastic_bag":7, "Battery":8, "Clothing":9}

classes_invert = {0:"General_trash", 1:"Paper", 2:"Paper_pack", 3:"Metal", 4:"Glass", 
           5:"Plastic", 6:"Styrofoam", 7:"Plastic_bag", 8:"Battery", 9:"Clothing"}
for i in classes.keys():
    os.makedirs(os.path.join(dataset_base,i),exist_ok=True)
    
real_json = {}
real_json['info'] = train_json['info']
real_json['licenses'] = train_json['licenses']
real_json['annotations'] = train_json['annotations']
real_json['categories'] = train_json['categories']
real_json['images'] = []

for i in train_json['images']:
    img_name = i['file_name'].split('/')[1]
    i['file_name'] = 'train_collage_background/'+img_name
    real_json['images'].append(i)




################## annotation category별로 잘라서 저장




for i in train_json['annotations']:
    # print('dataset/train/'+str(image_id).zfill(4)+'.jpg')
    image_id = i['image_id']
    ann_id = i['id']
    category = classes_invert.get(int(i['category_id']))
    x = int(i['bbox'][0])
    y = int(i['bbox'][1])
    w = math.ceil(i['bbox'][2])
    h = math.ceil(i['bbox'][3])
    
    img = cv2.imread('dataset/train/'+str(image_id).zfill(4)+'.jpg')
    cropped_img = img[y: y + h, x: x + w]
    
    cv2.imwrite('dataset/train_collage/'+category+'/'+str(ann_id)+'.jpg',cropped_img)
    
print("img crop done!")

