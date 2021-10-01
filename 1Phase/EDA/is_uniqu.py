import json
from collections import Counter

jf = '/opt/ml/detection/object-detection-level2-cv-03/1Phase/EDA/aug/train_aug.json'
# jf = '/opt/ml/detection/dataset/train.json'
train_aug = json.load(open(jf))
annots = train_aug['annotations']

an_ids = []
for ann in annots:
    an_ids.append(ann['id'])

# print(sorted(an_ids))
# print(len(an_ids))

cnt = Counter(an_ids)
# an_ids = sorted(an_ids)
# for idx, an_id in enumerate(an_ids):
#     print(f'{idx: <6}: {an_id}')

for k, v in cnt.items():
    if v != 1:
        print(k, v)