import os
import json
from glob import glob
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

confidence_thres = 0
dataroot = '/opt/ml/detection/dataset'
dstroot = '/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/work_dirs/swinB_Rmc5121024_4c1f_fullAugs+_defualtDS_LB661/ensembles/vis'
dstroot_crop = f'/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/work_dirs/swinB_Rmc5121024_4c1f_fullAugs+_defualtDS_LB661/ensembles/cutmixsrc'
inferencefile = '/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/work_dirs/swinB_5121024_LB661.csv'

infer_json = pd.read_csv(inferencefile)

imId_json = infer_json['image_id']
predictions_json = infer_json['PredictionString']


if not os.path.isdir(dstroot) and confidence_thres == 0:
    os.makedirs(dstroot)
if not os.path.isdir(dstroot_crop) and confidence_thres > 0:
    os.makedirs(dstroot_crop)

color_group = {'General trash': 'orange', 'Paper': 'white', 'Paper pack': 'ivory', 'Metal': 'dimgrey', 'Glass': 'dodgerBlue', 'Plastic': 'darkolivegreen', 'Styrofoam': 'khaki', 'Plastic bag': 'Teal', 'Battery': 'lime', 'Clothing': 'fuchsia'}
cls2name = list(color_group.keys())
color_group = list(color_group.values())
font = ImageFont.truetype('/opt/ml/detection/object-detection-level2-cv-03/1Phase/EDA/ubuntu.regular.ttf', 15)


pred_dict = {}
for imId in imId_json:
    pred_dict[imId] = []
    
for imId, predictions in zip(imId_json, predictions_json):
    if pd.isna(predictions):
        print(f'pd.isna: {predictions}')
        print('-'*30)
    else:
        predictions = str(predictions).split()
        for idx in range(0, len(predictions), 6):
            cls = predictions[idx]
            confidence = float(predictions[idx+1])
            pt1 = (float(predictions[idx+2]), float(predictions[idx+3]))
            pt2 = (float(predictions[idx+4]), float(predictions[idx+5]))
            pred_dict[imId].append([cls, confidence, pt1, pt2])


if confidence_thres == 0:
    print('1. inference visualizing')
    for imId,  preds in tqdm(pred_dict.items()):
        img = Image.open(os.path.join(dataroot, imId))
        draw = ImageDraw.Draw(img)
        for cls, confidence, pt1, pt2 in preds:
            draw.rectangle((pt1, pt2), outline=color_group[int(cls)], width=4)
            draw.text(pt1, f'{cls2name[int(cls)]}, {confidence: .4f}', (255,255,255), font=font, backrec=True)
            
        img.save(os.path.join(dstroot, imId.split('/')[-1]))

else:
    print('2. Image cropping for pseudo labeling')
    for imId, preds in tqdm(pred_dict.items()):
        img = Image.open(os.path.join(dataroot, imId))
        for idx, (cls, confidence, pt1, pt2) in enumerate(preds):
            if confidence >= confidence_thres:
                crs = img.crop((*pt1, *pt2))
                # draw = ImageDraw.Draw(crs)
                # draw.text((10,10), f'{cls2name[int(cls)]}, {confidence: .4f}', (255,255,255), font=font, backrec=True)
                crs.save(os.path.join(dstroot_crop, imId.split('/')[-1].split('.')[-2]+f'_{idx}_{cls}'+'.jpg'))

