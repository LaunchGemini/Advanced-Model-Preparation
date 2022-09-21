# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import random
import time
import json
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv import Config, ConfigDict
from mmcv.runner import CheckpointLoader

from mpa.utils.config_utils import MPAConfig, update_or_add_custom_hook
from mpa.utils.logger import config_logger, get_logger

from .registry import STAGES

logger = get_logger()


def _set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f'Training seed was set to {seed} w/ deterministic={deterministic}.')
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_available_types():
    types = []
    for k, v in STAGES.module_dict.items():
        # logger.info(f'key [{k}] = value[{v}]')
        types.append(