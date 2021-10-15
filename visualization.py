import os
import cv2
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv_path', type=str, default='//opt/ml/detection/UniverseNet/work_dirs/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_test_vote/submission_latest.csv')
parser.add_argument('--root_dir', type=str,  default='/opt/ml/detection/dataset')
parser.add_argument('--save_path', type=str, default='/opt/ml/detection/UniverseNet/work_dirs/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_test_vote')
parser.add_argument('--num_imgs', type=int, default=10)

if __name__ == '__main__':
    args = parser.parse_args()
    
    csv_path = args.csv_path
    root_dir = args.root_dir
    save_path = args.save_path
    classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
    COLORS = [
        (39, 129, 113), 
        (164, 80, 133), 
        (83, 122, 114), 
        (99, 81, 172), 
        (95, 56, 104), 
        (37, 84, 86), 
        (14, 89, 122),
        (80, 7, 65), 
        (10, 102, 25), 
        (90, 185, 109),
        (106, 110, 132)
    ]
    
    os.makedirs(save_path, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    image_id = df.image_id
    prediction_string = df.PredictionString
    
    for idx in tqdm(range(args.num_imgs)):
        img_path = os.path.join(root_dir, image_id[idx])
        img = cv2.imread(img_path)
        annot = prediction_string[idx].split(' ')
        annot = [annot[i * 6:(i + 1) * 6] for i in range((len(annot) + 6 - 1) // 6 )] 
        annot.remove([''])
        output_img = img.copy()
        for bbox in annot:
            bbox = list(map(float, bbox))
            bbox = np.array(bbox).astype(int)
            label = int(bbox[0])
            xmin, ymin, xmax, ymax = bbox[2:6]

            color = COLORS[label]
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)        
            text_size = cv2.getTextSize(classes[label], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 2, ymin + text_size[1] + 6), color, -1)
            cv2.putText(
                output_img, classes[label],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_ITALIC, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)        
        result_img = np.concatenate((img, output_img), axis=1) 
        
        cv2.imwrite(os.path.join(save_path, image_id[idx].replace('/', '-')), result_img)