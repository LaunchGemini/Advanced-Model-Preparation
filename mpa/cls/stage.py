# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import torch
import numpy as np

from mmcv import ConfigDict
from mmcv import build_from_cfg

from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger

logger = get_logger()

CLASS_INC_DATASET = ['MPAClsDataset', 'MPAMultilabelClsDataset', 'MPAHierarchicalClsDataset',
                     'ClsDirDataset', 'ClsTVDataset']
PSEUDO_LABEL_ENABLE_DATASET = ['ClassIncDataset', 'LwfTaskIncDataset', 'ClsTVDataset']
WEIGHT_MIX_CLASSIFIER = ['SAMImageClassifier']


class ClsStage(Stage):
    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        logger.info(f'configure: training={training}')

        # Recipe + model
        cfg = self.cfg
        if model_cfg:
            if hasattr(cfg, 'model'):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                cfg.model = copy.deepcopy(model_cfg.model)

        if cfg.model.pop('task', None) != 'classification':
            raise ValueError(
                f'Given model_cfg ({model_cfg.filename}) is not supported by classification recipe'
            )

        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)

        if cfg.get('resume', False):
            cfg.resume_from = cfg.load_from

        self.configure_model(cfg, training, **kwargs)

        # OMZ-plugin
        if cfg.model.backbone.type == 'OmzBackboneCls':
            ir_path = kwargs.get('ir_path', None)
            if ir_path is None:
                raise RuntimeError('OMZ model needs OpenVINO bin/XML files.')
            cfg.model.backbone.model_path = ir_path

        pretrained = kwargs.get('pretrained', None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)
        self.configure_data(cfg, training, **kwargs)

        # Task
        if 'task_adapt' in cfg:
            model_meta = self.get_model_meta(cfg)
            model_tasks, dst_classes = self.configure_task(cfg, training, model_meta, **kwargs)
            if model_tasks is not None:
                self.model_tasks = model_tasks
            if dst_classes is not None:
                self.model_classes = dst_classes
        else:
            if 'num_classes' not in cfg.data:
                cfg.data.num_classes = len(cfg.data.tra