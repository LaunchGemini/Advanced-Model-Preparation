# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch

from mmcv import ConfigDict
from mmdet.datasets import build_dataset
from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger

logger = get_logger()


class DetectionStage(Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        logger.info(f'configure!: training={training}')

        # Recipe + model
        cfg = self.cfg
        if model_cfg:
            if hasattr(model_cfg, 'model'):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                raise ValueError("Unexpected config was passed through 'model_cfg'. "
                                 "it should have 'model' attribute in the config")
            model_task = cfg.model.pop('task', 'detection')
            if model_task != 'detection':
                raise ValueError(
                    f'Given model_cfg ({model_cfg.filename}) is not supported by detection recipe'
                )
        self.configure_model(cfg, training, **kwargs)

        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        pretrained = kwargs.get('pretrained', None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained  # Overriding by stage input

        if cfg.get('resume', False):
            cfg.resume_from = cfg.load_from

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)
        self.configure_data(cfg, training, **kwargs)

        # Task
        if 'task_adapt' in cfg:
            self.configure_task(cfg, training, **kwargs)

        # Regularization
        if training:
            self.configure_regularization(cfg)

        # Hooks
        self.configure_hook(cfg)

        return cfg

    def configure_model(self, cfg, training, **kwargs):
        super_type = cfg.model.pop('super_type', None)
        if super_type:
            cfg.model.arch_type = cfg.model.type
            cfg.model.type = super_type

        # OMZ-plugin
        if cfg.model.backbone.type == 'OmzBackboneDet':
            ir_path = kwargs.get('ir_path')
            if not ir_path:
                raise RuntimeError('OMZ model needs OpenVINO bin/xml files.')
            cfg.model.backbone.model_path = ir_path
            if cfg.model.type == 'SingleStageDetector':
                cfg.model.bbox_head.model_path = ir_path
            elif cfg.model.type == 'FasterRCNN':
                cfg.model.rpn_head.model_path = ir_path
            else:
                raise NotImplementedError(f'Unknown model type - {cfg.model.type}')

    def configure_anchor(self, cfg, proposal_ratio=None):
        if cfg.model.type in ['SingleStageDetector', 'CustomSingleStageDetector']:
            anchor_cfg = cfg.model.bbox_head.anchor_generator
            if anchor_cfg.type == 'SSDAnchorGeneratorClustered':
                cfg.model.bbox_head.anchor_generator.pop('input_size', None)

    def configure_data(self, cfg, training, **kwargs):
        Stage.configure_data(cfg, training, **kwargs)
        super_type = cfg.data.train.pop('super_type', None)
        if super_type:
            cfg.data.train.org_type = cfg.data.train.type
            cfg.data.train.type = super_type
        if training:
            if 'unlabeled' in cfg.data and cfg.data.unlabeled.get('img_file', None):
                cfg.data.unlabeled.ann_file = cfg.data.unlabeled.pop('img_file')
                if len(cfg.dat