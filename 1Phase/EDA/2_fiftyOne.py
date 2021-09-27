import fiftyone as fo
import fiftyone.zoo as foz

data_path = '/opt/ml/detection/dataset/train'
labels_path = '/opt/ml/detection/dataset/train.json'

dataset = fo.Dataset.from_dir(
	dataset_type = fo.types.COCODetectionDataset,
	data_path = data_path,
	labels_path = labels_path
	)

session = fo.launch_app(dataset)
