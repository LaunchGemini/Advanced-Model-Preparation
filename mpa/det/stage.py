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
                if len(cfg.data.unlabeled.get('pipeline', [])) == 0:
                    cfg.data.unlabeled.pipeline = cfg.data.train.pipeline.copy()
                update_or_add_custom_hook(
                    cfg,
                    ConfigDict(
                        type='UnlabeledDataHook',
                        unlabeled_data_cfg=cfg.data.unlabeled,
                        samples_per_gpu=cfg.data.unlabeled.pop('samples_per_gpu', cfg.data.samples_per_gpu),
                        workers_per_gpu=cfg.data.unlabeled.pop('workers_per_gpu', cfg.data.workers_per_gpu),
                        seed=cfg.seed
                    )
                )
        for subset in ("train", "val", "test"):
            if 'dataset' in cfg.data[subset]:
                subset_cfg = self.get_data_cfg(cfg, subset)
                subset_cfg.ote_dataset = cfg.data[subset].pop('ote_dataset', None)
                subset_cfg.labels = cfg.data[subset].get('labels', None)
                if 'data_classes' in cfg.data[subset]:
                    subset_cfg.data_classes = cfg.data[subset].pop('data_classes')
                if 'new_classes' in cfg.data[subset]:
                    subset_cfg.new_classes = cfg.data[subset].pop('new_classes')

    def configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation
        """
        logger.info(f'task config!!!!: training={training}')
        task_adapt_type = cfg['task_adapt'].get('type', None)
        task_adapt_op = cfg['task_adapt'].get('op', 'REPLACE')

        # Task classes
        org_model_classes, model_classes, data_classes = \
            self.configure_task_classes(cfg, task_adapt_type, task_adapt_op)

        # Data pipeline
        if data_classes != model_classes:
            self.configure_task_data_pipeline(cfg, model_classes, data_classes)

        # Evaluation dataset
        if cfg.get('task', 'detection') == 'detection':
            self.configure_task_eval_dataset(cfg, model_classes)

        # Training hook for task adaptation
        self.configure_task_adapt_hook(cfg, org_model_classes, model_classes)

        # Anchor setting
        if cfg['task_adapt'].get('use_mpa_anchor', False):
            self.configure_anchor(cfg)

        # Incremental learning
        self.configure_task_cls_incr(cfg, task_adapt_type, org_model_classes, model_classes)

    def configure_task_classes(self, cfg, task_adapt_type, task_adapt_op):

        # Input classes
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)

        # Model classes
        if task_adapt_op == 'REPLACE':
            if len(data_classes) == 0:
                model_classes = org_model_classes.copy()
            else:
                model_classes = data_classes.copy()
        elif task_adapt_op == 'MERGE':
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f'{task_adapt_op} is not supported for task_adapt options!')

        if task_adapt_type == 'mpa':
            data_classes = model_classes
        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
      