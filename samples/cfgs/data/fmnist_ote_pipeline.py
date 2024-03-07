_base_ = ["./pipelines/fixmatch_pipeline.py"]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

seed = 1234

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    num_classes=10,
    train=dict(
