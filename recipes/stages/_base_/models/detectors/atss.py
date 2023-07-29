_base_ = './detector.py'

model = dict(
    type='ATSS',
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=64,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True
    ),
    