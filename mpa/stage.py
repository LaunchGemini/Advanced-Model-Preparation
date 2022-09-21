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
        types.append(k)
    return types


# @STAGES.register_module()
class Stage(object):
    def __init__(self, name, mode, config, common_cfg={}, index=0, **kwargs):
        logger.debug(f'init stage with: {name}, {mode}, {config}, {common_cfg}, {index}, {kwargs}')
        # the name of 'config' cannot be changed to such as 'config_file'
        # because it is defined as 'config' in recipe file.....
        self.name = name
        self.mode = mode
        self.index = index
        self.input = kwargs.pop('input', {})  # input_map?? input_dict? just input?
        self.output_keys = kwargs.pop('output', [])

        if common_cfg is None:
            common_cfg = dict(output_path='logs')

        if not isinstance(common_cfg, dict):
            raise TypeError(f'common_cfg should be the type of dict but {type(common_cfg)}')
        else:
            if common_cfg.get('output_path') is None:
                logger.info("output_path is not set in common_cfg. set it to 'logs' as default")
                common_cfg['output_path'] = 'logs'

        self.output_prefix = common_cfg['output_path']
        self.output_suffix = f'stage{self.index:02d}_{self.name}'

        # # Work directory
        # work_dir = os.path.join(self.output_prefix, self.output_suffix)
        # mmcv.mkdir_or_exist(os.path.abspath(work_dir))

        if isinstance(config, Config):
            cfg = config
        elif isinstance(config, dict):
            cfg = Config(cfg_dict=config)
        elif isinstance(config, str):
            if os.path.exists(config):
                cfg = MPAConfig.fromfile(config)
            else:
                err_msg = f'cannot find configuration file {config}'
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            err_msg = "'config' argument could be one of the \
                       [dictionary, Config object, or string of the cfg file path]"
            logger.error(err_msg)
            raise ValueError(err_msg)

        cfg.merge_from_dict(common_cfg)

        if len(kwargs) > 0:
            addtional_dict = {}
            logger.info('found override configurations for the stage')
            for k, v in kwargs.items():
                addtional_dict[k] = v
                logger.info(f'\t{k}: {v}')
            cfg.merge_from_dict(addtional_dict)

        max_epochs = -1
        if hasattr(cfg, 'total_epochs'):
            max_epochs = cfg.pop('total_epochs')
        if hasattr(cfg, 'runner'):
            if hasattr(cfg.runner, 'max_epochs'):
                if max_epochs != -1:
                    max_epochs = min(max_epochs, cfg.runner.max_epochs)
                else:
                    max_epochs = cfg.runner.max_epochs
        if max_epochs > 0:
            if cfg.runner.max_epochs != max_epochs:
                cfg.runner.max_epochs = max_epochs
                logger.info(f'The maximum number of epochs is adjusted to {max_epochs}.')
            if hasattr(cfg, 'checkpoint_config'):
                if hasattr(cfg.checkpoint_config, 'interval'):
                    if cfg.checkpoint_config.interval > max_epochs:
                        logger.warning(f'adjusted checkpoint interval from {cfg.checkpoint_config.interval} to 