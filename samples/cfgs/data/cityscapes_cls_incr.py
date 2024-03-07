_base_ = [
    './pipelines/cls_incr_cityscapes.py'
]

__dataset_type = 'CityscapesDataset'
__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

seed = 1234

data = dict(
    samples_per_gpu=8