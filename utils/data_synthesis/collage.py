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




#기존 train.json에서 annotation & image 몇개 있었는지 prefix
annotations_prefix = 23143
count_filename = 4883
##################
null = None

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# image_prefix = 5000
collage_path = 'dataset/train_collage/'

General_list = os.listdir('dataset/train_collage/General_trash')
if '.ipynb_checkpoints' in General_list:
        General_list.remove('.ipynb_checkpoints')
        
paper_list = os.listdir('dataset/train_collage/Paper')
if '.ipynb_checkpoints' in paper_list:
        paper_list.remove('.ipynb_checkpoints')
paperpack_list = os.listdir('dataset/train_collage/Paper_pack')
if '.ipynb_checkpoints' in paperpack_list:
        paperpack_list.remove('.ipynb_checkpoints')
metal_list = os.listdir('dataset/train_collage/Metal')
if '.ipynb_checkpoints' in metal_list:
        metal_list.remove('.ipynb_checkpoints')
glass_list = os.listdir('dataset/train_collage/Glass')
if '.ipynb_checkpoints' in glass_list:
        glass_list.remove('.ipynb_checkpoints')
plastic_list = os.listdir('dataset/train_collage/Plastic')
if '.ipynb_checkpoints' in plastic_list:
        plastic_list.remove('.ipynb_checkpoints')
styrofoam_list = os.listdir('dataset/train_collage/Styrofoam')
if '.ipynb_checkpoints' in styrofoam_list:
        styrofoam_list.remove('.ipynb_checkpoints')
plasticbag_list = os.listdir('dataset/train_collage/Plastic_bag')
if '.ipynb_checkpoints' in plasticbag_list:
        plasticbag_list.remove('.ipynb_checkpoints')
battery_list = os.listdir('dataset/train_collage/Battery')
if '.ipynb_checkpoints' in battery_list:
        battery_list.remove('.ipynb_checkpoints')
clothing_list = os.listdir('dataset/train_collage/Clothing')
if '.ipynb_checkpoints' in clothing_list:
        clothing_list.remove('.ipynb_checkpoints')
bg_list = os.listdir('dataset/BG')
if '.ipynb_checkpoints' in bg_list:
        bg_list.remove('.ipynb_checkpoints')
        

random.shuffle(General_list)
random.shuffle(paper_list)
random.shuffle(paperpack_list)
random.shuffle(metal_list)
random.shuffle(glass_list)
random.shuffle(plastic_list)
random.shuffle(styrofoam_list)
random.shuffle(plasticbag_list)
random.shuffle(battery_list)
random.shuffle(clothing_list)

comb = combinations(["General_trash", "Paper", "Paper_pack", "Metal","Paper", "Paper_pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic_bag", "Clothing","Battery","Battery","Battery"], 4)
listComb = list(comb)
print("Combination 개수",len(listComb))
masterList = []
for j in range(0,1100): #몇장만들지

    
    random.shuffle(listComb)
    cnt = 0
    for c in listComb:
        new_items = []
        listC = list(c)
        # print(listC)
        random.shuffle(listC)
        # print(listC)
        for x in listC:
            if x == 'General_trash':
                i = random.randint(0, 3965)
                general = collage_path + 'General_trash/'+General_list[i]
                new_items.append(general)
            if x == 'Paper':
                i = random.randint(0, 6351)
                paper = collage_path + 'Paper/'+paper_list[i]
                new_items.append(paper)
            if x == 'Paper_pack':
                i = random.randint(0, 896)
                paper_pack = collage_path + 'Paper_pack/'+paperpack_list[i]
                new_items.append(paper_pack)
            if x == 'Metal':
                i = random.randint(0, 935)
                metal = collage_path + 'Metal/'+metal_list[i]
                new_items.append(metal)
            if x == 'Glass':
                i = random.randint(0, 981)
                glass = collage_path + 'Glass/'+glass_list[i]
                new_items.append(glass)
            if x == 'Styrofoam':
                i = random.randint(0, 1262)
                styrofoam = collage_path + 'Styrofoam/'+styrofoam_list[i]
                new_items.append(styrofoam)
            if x == 'Plastic_bag':
                i = random.randint(0, 5177)
                plastic_bag = collage_path + 'Plastic_bag/'+plasticbag_list[i]
                new_items.append(plastic_bag)
            if x == 'Battery':
                i = random.randint(0, 158)
                battery = collage_path + 'Battery/'+battery_list[i]
                new_items.append(battery)
            if x == 'Clothing':
                i = random.randint(0, 467)
                clothing = collage_path + 'Clothing/'+clothing_list[i]
                new_items.append(clothing)
            if x == 'Plastic':
                i = random.randint(0, 2942)
                plastic = collage_path + 'Plastic/'+plastic_list[i]
                new_items.append(plastic)
        masterList.append(new_items)
        cnt+=1
        if cnt==1:
            break
print(len(masterList))

for ll in masterList:
    # #white canvas
    im = Image.new('RGB', (1024, 1024), 'white')
    im.format = "JPG"
    
    #background canvas
    bg_path = 'dataset/BG/'+ bg_list[j]
    im = Image.open(bg_path)
    im.format = "JPG"
    
    filename = str(count_filename)+'.jpg'
    # print(filename)
    # A = [os.path.join('dataset/train',
                      # list(split_text(x))[0].replace('.png', '') + '_cropped/' + x+'.png') for x in ll]
    count = 0
    xoff = 15
    yoff = 15
    coordinates = ((0,0),(512,0),(0, 512),(512, 512))
    # coordinates = ((0,0),(512,0),(0, 512),(512, 512))
    dictObjects = {}
    for img in ll:
        x,y = coordinates[count]
        x += random.randint(0,20)
        y += random.randint(0,20)
        imgg = Image.open(img)
        # object_name = list(split_text(img.split('/')[-1]))[0]
        object_name = img.split('/')[2]
        # bbox = imgg.convert("RGBa").getbbox()
        w, h = imgg.size
        bbox = [x,y,w,h]
        if x+w >= 1024 or y+h >=1024:
            # print(bbox)
            continue
        draw = ImageDraw.Draw(im)
        
        im.paste(imgg, (x,y))
        count = count + 1
        #########################
        annotations_prefix+=1
        
        #####json
        
        row = {'image_id':count_filename,
        'category_id': classes[object_name],
        'area':w*h,
        'bbox':bbox,
        'iscrowd':0,
        'id':annotations_prefix}
        
        real_json['annotations'].append(row)
        
        
    filepath = os.path.join('dataset/train_collage_background',filename)
    im.save(filepath)
    row_img = {'width':1024,
    'height':1024,
    'file_name':"train_collage_background/"+filename,
    'license':0,
    'flickr_url': null,
    'coco_url':null,
    'data_captured':"2020-12-26 14:44:23",
    'id':count_filename}
    real_json['images'].append(row_img)
    count_filename +=1
    
#json COCO format으로 save   
with open("train_collage_background.json", "w") as f: 
    json.dump(real_json, f)

  
print("FIN")

    
