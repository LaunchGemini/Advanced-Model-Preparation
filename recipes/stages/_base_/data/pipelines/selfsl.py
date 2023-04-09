__img_norm_cfg = dict(mean=None, std=None)
__resize_target_size = -1

train_pipeline_v0 = [
    dict(type='RandomResizedCrop', size=__resize_target_size),
    dict(type='RandomHoriz