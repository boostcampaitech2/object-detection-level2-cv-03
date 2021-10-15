import pandas as pd
import numpy as np
from ensemble_boxes import *
from tqdm import tqdm

# image_id : image id
# PredictionString : label, score, x(min), y(min), x(max), y(max)
# sample=pd.read_csv('/opt/ml/myfolder/UniverseNet-master/submissions/universe101_fold0_5759.csv')
sample=pd.read_csv('/opt/ml/detection/UniverseNet/work_dirs/univ101_default/swinL.csv')
image_list=sample['image_id']

# LB_weights = np.array([ 0.603, 0.661, 0.636, 0.585])
# LB_weights = np.array([0.566, 0.677])
# LB_weights = list(LB_weights / np.sum(LB_weights))
LB_weights = None
dstroot = '/opt/ml/detection/UniverseNet/univ_fold'

# .585
# .603 
# .543
# .637


df_preds=[
    pd.read_csv('/opt/ml/detection/UniverseNet/univ_fold/universe101_kfold1.csv'),
    pd.read_csv("/opt/ml/detection/UniverseNet/univ_fold/universe101_kfold2.csv"),
    pd.read_csv("/opt/ml/detection/UniverseNet/univ_fold/universe101_kfold3.csv"),
    pd.read_csv("/opt/ml/detection/UniverseNet/univ_fold/universe101_kfold5.csv"),

    # pd.read_csv("/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/work_dirs/swinB_Rmc5121024_4c1f_fullAugs+_defualtDS_LB661/submissions/zero_thres/B_anchor_zeroThres_LB653.csv"),
    # pd.read_csv("/opt/ml/detection/object-detection-level2-cv-03/1Phase/mmdetection/work_dirs/swinB_mc512864_fullAugs_defualtDS/swinT_tta_LB585.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/swin_s_basic_fold0.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/swins_6013_all_data.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/swint_allcustomdata_5712.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/SwinT_epoch50_resize512_LB5543.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/swint_kfold_5732.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/universe101_fold2_5559.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/universe101_swa_fold0_5769.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/universe101_swa_fold1_5726.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/universe101_swa_fold3_5727.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/universe101_swa_fold4_5767.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/universe101_swa_kfold_0.6102.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/universe50_5472.csv"),
    # pd.read_csv("/opt/ml/myfolder/UniverseNet-master/work_dirs/all_csv/vfnetx_alldata_swa_5558.csv"),

]
model_count = len(df_preds)

imsize = 1024.

def get_reference(df_pred, idx):
    pred_list = df_pred['PredictionString'].tolist()
    ref = pred_list[idx]
    return ref

def get_model_pred(df_pred):
    pred_list = df_pred['PredictionString'].tolist()
    result=[]
    for idx, image_pred in enumerate(pred_list):
        if pd.isna(image_pred):
            print('hi')
            image_pred = get_reference(df_preds[0], idx)
            # image_pred = '0 0 0 0 0 0'
        pred_line=image_pred.split(' ')[:-1]
        pred_line=np.array(pred_line).reshape(-1,6).transpose()
        boxes_count=pred_line.shape[1]
        labels=pred_line[0].astype(np.int32)
        scores=pred_line[1].astype(np.float32)
        boxes=[]
        for idx in range(boxes_count):
            pred_boxes=[]
            for loc in range(2,6):
                pred_boxes.append(float(pred_line[loc][idx]))
            boxes.append(pred_boxes)

        boxes = np.array(boxes)/imsize

        result.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,})
    return result

def box_size(boxes):
    if ((boxes[2]-boxes[0])*(boxes[3]-boxes[1])*imsize )> (32*32):
        return True
    else :
        return False
                

    return (boxes[2]-boxes[0])*(boxes[3]-boxes[1])
model_prediction=[]
for df_pred in df_preds:
    model_prediction.append(get_model_pred(df_pred))


def run_wbf(prediction, model_count, image_index, iou_thr=0.5, skip_box_thr=0.05, weights=None):
    scores=[]
        # print('boxsize : ',box_size(prediction[model_idx][image_index]['boxes']))
    # for model_idx in range(0,model_count):
    #     tmp2=[]
    #     for model_idx2 in range(0,len(prediction[model_idx][image_index]['boxes'])):
    #         tmp=[]
    #         if model_idx==1:#model 큰 거
    #             if box_size(prediction[model_idx][image_index]['boxes'][model_idx2]):
    #                 tmp2.append(prediction[model_idx][image_index]['scores'][model_idx2])
    #             else :
    #                 tmp2.append(0)
    #         else : # model 작은 거 
    #             if box_size(prediction[model_idx][image_index]['boxes'][model_idx2]):
    #                 tmp2.append(0)
    #             else :
    #                 tmp2.append(prediction[model_idx][image_index]['scores'][model_idx2])
    #     scores.append(tmp2)
    boxes = [prediction[model_idx][image_index]['boxes'].tolist() for model_idx in range(0,model_count)]
    scores = [prediction[model_idx][image_index]['scores'].tolist() for model_idx in range(0,model_count)]
    labels = [prediction[model_idx][image_index]['labels'].tolist() for model_idx in range(0,model_count)]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*imsize
    return boxes, scores, labels

prediction_strings = []
file_names = []
for index, image_id in enumerate(tqdm(image_list)):
    boxes, scores, labels = run_wbf(model_prediction,model_count,index, iou_thr=0.6, skip_box_thr=0.03, weights=LB_weights)
    
    prediction_string = ''
    for box, score, label in zip(boxes, scores, labels):
        # if score > 0.95:
        #     # print(score)
        #     score = 1.00
        prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
            box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
    prediction_strings.append(prediction_string)
    file_names.append(image_id)


submission = pd.DataFrame()
submission['image_id'] = file_names
submission['PredictionString'] = prediction_strings
# submission.to_csv(f'{dstroot}/B_B+pse_Univ.csv', index=None)
# submission.to_csv(f'{dstroot}/T_B_B+pse_Univ_VFNt.csv', index=None)
# submission.to_csv(f'{dstroot}/zeroTH_BBU.csv', index=None)
# submission.to_csv(f'{dstroot}/zeroTH_TBBU.csv', index=None)
submission.to_csv(f'{dstroot}/univ_fold_0_03.csv', index=None)
print(submission.head())