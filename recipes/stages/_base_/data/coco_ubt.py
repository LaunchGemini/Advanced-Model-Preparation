_base_ = [
    './data.py',
    './pipelines/ubt.py',
]

__dataset_type = 'CocoDataset'
__data_root_path = 'data/coco/'

__train_pipeline = {{_base_.train_pipeline}}
# __unlabeled_pipeline = __train_pipeline.copy().pop(1)  # Removing 'LoadAnnotations' op
__unlabeled_pipeline = {{_base_.unlabeled_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__samples_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        super_type='PseudoBalancedDataset',
        ann_file=__data_root_path + 'annotations/instances_train2017.json',
        img_prefix=__data_root_path + 'train2017/',
        pipeline=__train_pipeline,
        classes=[
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truckl',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich