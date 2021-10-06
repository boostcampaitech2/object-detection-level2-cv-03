import numpy as np
import random
#import augmixations
from .augmixations.cutmix import SmartCutmix
#from augmixations.augmixations.cutmix import SmartCutmix
from mmdet.datasets import build_dataset
from mmcv import Config
from mmdet.datasets import PIPELINES

dataset_type = 'CocoDataset'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
num_classes = len(classes)
root='/opt/ml/detection/dataset/'



@PIPELINES.register_module()
class CustomCutmix:
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self,
        dataset_type = dataset_type,
        classes=classes,
        ann_file=root+'kfold/cv_train1.json',
        img_prefix=root,
        p=0.5,
        T=30):

        self.p = p
        self.data_cfg = Config(
            dict(
                train=dict(
                    type=dataset_type,
                    classes=classes,
                    ann_file=ann_file,
                    img_prefix=root)))
        
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)]
        self.data_cfg.train['pipeline'] = train_pipeline
    
        self.datasets = build_dataset(self.data_cfg.train) 
        self.img2idx = { img_id : idx for idx,img_id in enumerate(self.datasets.img_ids)}
        self.label2anno=[ [] for _ in range(num_classes)]
        for idx in self.datasets.coco.anns.keys():
            label = self.datasets.coco.anns[idx]['category_id']
            self.label2anno[label].append(self.datasets.coco.anns[idx])
        
        # 분포에 따라 weight를 준 prob 구하기
        instance_count = [len(label_count) for label_count in self.label2anno ]
        normedWeights = [1 - (x / sum(instance_count)) for x in instance_count]
        self.label_probs = [ np.exp(x*T)/ np.exp(np.array(normedWeights)*T).sum() for x in normedWeights]

    def __call__(self, results):
        if random.random() < self.p:

            img =results['img']
            gt_bboxes = results['gt_bboxes']
            gt_labels = results['gt_labels']
            
            # 1. cutmix할 object의 label값 선택
            label =np.random.choice(list(range(num_classes)),p=self.label_probs)
            # 2. 해당 label의 annotation 정보를 랜덤으로 선택
            ann = np.random.choice(self.label2anno[label])
            # 3. annotation을 포함한 image id를 구하고 해당 image id를 dataset index로 변환
            cut_idx = self.img2idx[ann['image_id']]

            cut_data = self.datasets.prepare_train_img(cut_idx)            
            cut_img = cut_data['img']
            cut_bboxes = cut_data['gt_bboxes']
            cut_labels = cut_data['gt_labels']

            crop_rect_config = {
                'crop_x' : int(ann['bbox'][0]),
                'crop_y' : int(ann['bbox'][1]),
                'rect_w' : int(ann['bbox'][2]),  
                'rect_h' : int(ann['bbox'][3]),
                'insert_x' : None,
                'insert_y' : None,
            }
            process_boxes_config = {
                'max_overlap_area_ratio': 0.75,
                'min_height_result_ratio': 0.25,
                'min_width_result_ratio': 0.25,
                'max_height_intersection': 0.9,
                'max_width_intersection': 0.9,
                }

            self.cutmix = SmartCutmix(crop_rect_config,process_boxes_config)
            
            new_img, new_boxes, new_labels = self.cutmix(
                img,gt_bboxes,gt_labels,
                cut_img,cut_bboxes,cut_labels
                )
            
            results['img'] = new_img
            results['gt_bboxes'] = new_boxes
            results['gt_labels'] = new_labels
        return results
    