_base_ = [
    './datasets/pipelines/rcrop_hflip_resize.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

