
_base_ = './detector.py'

model = dict(
    type='YOLOX',
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.375),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=80,
        in_channels=96,
        feat_channels=96),
    t