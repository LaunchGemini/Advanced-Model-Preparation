# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook
from torch.utils.data import DataLoader

from mpa.modules.datasets.samplers.cls_incr_sampler import ClsIncrSampler
from mpa.modules.datasets.samplers.balanced_sampler import BalancedSampler
from mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class TaskAdaptHook(Hook):
    """Task Adaptation Hook for Task-Inc & Class-Inc

    Args:
        src_classes (list): A list of old classes used in the existing model
        dst_classes (list): A list of classes including new_classes to be newly learned
        model_type (str): Types of models used for learning
        sampler_flag (bool): Flag about using ClsIncrSampler
        efficient_mode (bool): Flag about using efficient mode sampler
    """

    def __init__(self,
                 src_classes,
                 dst_classes,
                 model_type='FasterRCNN',
                 sampler_flag=False,
                 sampler_type='cls_incr',
                 efficient_mode=False):
        self.src_classes = src_classes
        self.dst_classes = dst_classes
        self.model_type = model_type
        self.sampler_flag = sampler_flag
        self.sampler_type = sampler_type
        self.efficient_mode = efficient_mode

        logger.info(f'Task Adaptation: {self.src_classes} => {self.dst_classes}')
        logger.info(f'- Efficient Mode: {self.efficient_mode}')
        logger.info(f'- Sampler type: {self.sampler_type}')
        logger.info(f'- Sampler flag: {self.sampler_flag}')

    def before_epoch(self, runner):
        if self.sampler_flag:
            dataset = runner.data_loader.dataset
            batch