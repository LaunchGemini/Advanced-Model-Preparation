_base_ = './single_stage_detector.py'

__width_mult = 1.0

model = dict(
    bbox_head=dict(
        type='SSDHead',
        num_c