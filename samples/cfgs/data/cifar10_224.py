_base_ = [
    './pipelines/hflip_resize.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    num_classes=10,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        Normalize=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
    ),
    train=dict(
        type='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=__train_pipeline
    ),
    test=dict(
        type='CIFAR10',
        data_prefix='data/torchvision/cifar10',
 