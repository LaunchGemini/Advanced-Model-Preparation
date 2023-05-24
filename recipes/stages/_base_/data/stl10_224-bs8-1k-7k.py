_base_ = [
    './datasets/types/tv_dataset_split.py',
    './datasets/pipelines/semisl_pipeline.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__trai