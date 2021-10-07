import json
import os

train_path = '/opt/ml/detection/Augmented/Pseudos/5495pse_B_B+pse_Univ_LB674/full_pseudo_cv_train1.json'
val_path = '/opt/ml/detection/Augmented/Pseudos/5495pse_B_B+pse_Univ_LB674/genuine/cv_val1.json'

train = json.load(open(train_path, 'r'))
val = json.load(open(val_path, 'r'))

train_images = train['images']
val_images = val['images']

train_set = set()
val_set = set()

for img in train_images:
    train_set.update([ img['file_name'] ])
for img in val_images:
    val_set.update([ img['file_name'] ])

print(train_set.intersection(val_set))
compromised = train_set.intersection(val_set)
print(f'isCompromised: {len(compromised)} / {len(val_set)}:  { len(compromised) / len(val_set) * 100: .2f}%')

print(f'train_images: {len(train_set)}')