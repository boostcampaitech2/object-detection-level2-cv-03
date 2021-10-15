import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np

# my_cfg = '/home/hci/Videos/object-detection-level2-cv-03/1Phase/mmdetection/configs/_base_/models/Swin/4VGA_cascade_swin.py'



classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기

# root='../dataset/'
root='/opt/ml/detection/dataset/'

############################ 수정하셔야하는 부분 ###############################
'''
제가 드린 config file 경로를 넣어주시면 됩니다. (ex )/opt/ml/detection/.../Univeresenet101.py
'''
my_cfg = '/opt/ml/CBNEt/UniverseNet/work_dirs/cbnet/cbnet.py'
###########################################################################


cfg = Config.fromfile(my_cfg)




############################ 수정하셔야하는 부분 ###############################

'''
cfg.work_dir은 최종 제출 파일 csv 이 저장되는 위치 입니다. 편하게 정해주시면 됩니다.
checkpoint_path는 각자 진행하신 최고 모델인 best_bbox_mAP_epoch{num}.pth 위치를 넣어주시면 됩니다.
'''
cfg.work_dir = '/opt/ml/CBNEt/UniverseNet/work_dirs'


checkpoint_path = '/opt/ml/CBNEt/UniverseNet/work_dirs/cbnet/best_bbox_mAP_epoch_39.pth'

fold_num = 1

###########################################################################

# dataset config 수정
cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json'
cfg.data.test.pipeline[1]['img_scale'] = [(x , x ) for x in range (512, 1280+1,128)] # Resize
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 2

cfg.seed=2021
cfg.gpu_ids = [0]

# cfg.model.roi_head.bbox_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None
cfg.model.test_cfg['nms'] = dict(type='soft_nms', iou_threshold=0.6)
    
# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# checkpoint path

model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, data_loader, show_score_thr=0.0) # output 계산

# submission 양식에 맞게 output 후처리
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.ann_file)
img_ids = coco.getImgIds()

class_num = 10
for i, out in enumerate(output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for j in range(class_num):
        for o in out[j]: #output class , confidence , x0 , y0, x1, y1 
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '
    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.work_dir, f'cbnet.csv'), index=None)
submission.head()