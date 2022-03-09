# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import numbers
import os.path as osp
import time
import warnings
import torch
import numpy as np
import random

import torch.multiprocessing as mp
import torch.distributed as dist

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Fp16OptimizerHook, build_optimizer, build_runner, HOOKS

from mmcls import __version__
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier
from mmcls.utils import collect_env
from mmcls.core import DistOptimizerHook

from mpa.registry import STAGES
from mpa.modules.datasets.composed_dataloader import ComposedDL
from mpa.stage import Stage
from mpa.cls.stage import ClsStage
from mpa.modules.hooks.eval_hook import CustomEvalHook, DistCustomEvalHook
from mpa.modules.hooks.fp16_sam_optimizer_hook import Fp16SAMOptimizerHook
from mpa.utils.logger import get_logger
from mpa.utils.data_cpu import MMDataCPU

logger = get_logger()


@STAGES.register_module()
class ClsTrainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=True, **kwargs)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        distributed = False
        if cfg.gpu_ids is not None:
            if isinstance(cfg.get('gpu_ids'), numbers.Number):
                cfg.gpu_ids = [cfg.get('gpu_ids')]
            if len(cfg.gpu_ids) > 1:
                distributed = True

        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)

        # Data
        if 'unlabeled' in cfg.data:
            datasets = [[build_dataset(cfg.data.train), build_dataset(cfg.data.unlabeled)]]
        else:
            datasets = [build_dataset(cfg.data.train)]

        # Dataset for HPO
        hp_config = kwargs.get('hp_config', None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDa