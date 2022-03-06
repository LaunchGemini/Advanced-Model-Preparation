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
                cfg.data.num_classes = len(cfg.data.train.get('classes', []))
            cfg.model.head.num_classes = cfg.data.num_classes

        if cfg.model.head.get('topk', False) and isinstance(cfg.model.head.topk, tuple):
            cfg.model.head.topk = (1,) if cfg.model.head.num_classes < 5 else (1, 5)
            if cfg.model.get('multilabel', False) or cfg.model.get('hierarchical', False):
                cfg.model.head.pop('topk', None)

        return cfg

    @staticmethod
    def configure_model(cfg, training, **kwargs):
        # verify and update model configurations
        # check whether in/out of the model layers require updating

        if cfg.get('load_from', None) and cfg.model.backbone.get('pretrained', None):
            cfg.model.backbone.pretrained = None

        update_required = False
        if cfg.model.get('neck') is not None:
            if cfg.model.neck.get('in_channels') is not None and cfg.model.neck.in_channels <= 0:
                update_required = True
        if not update_required and cfg.model.get('head') is not None:
            if cfg.model.head.get('in_channels') is not None and cfg.model.head.in_channels <= 0:
                update_required = True
        if not update_required:
            return

        # update model layer's in/out configuration
        input_shape = [3, 224, 224]
        logger.debug(f'input shape for backbone {input_shape}')
        from mmcls.models.builder import BACKBONES as backbone_reg
        layer = build_from_cfg(cfg.model.backbone, backbone_reg)
        output = layer(torch.rand([1] + input_shape))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        output = output.shape[1]
        if cfg.model.get('neck') is not None:
            if cfg.model.neck.get('in_channels') is not None:
                logger.info(f"'in_channels' config in model.neck is updated from "
                            f"{cfg.model.neck.in_channels} to {output}")
                cfg.model.neck.in_channels = output
                input_shape = [i for i in range(output)]
                logger.debug(f'input shape for neck {input_shape}')
                from mmcls.models.builder import NECKS as neck_reg
                layer = build_from_cfg(cfg.model.neck, neck_reg)
                output = layer(torch.rand([1] + input_shape))
                if isinstance(output, (tuple, list)):
                    output = output[-1]
                output = output.shape[1]
        if cfg.model.get('head') is not None:
            if cfg.model.head.get('in_channels') is not None:
                logger.info(f"'in_channels' config in model.head is updated from "
                            f"{cfg.model.head.in_channels} to {output}")
                cfg.model.head.in_channels = output

            # checking task incremental model configurations

    @staticmethod
    def configure_task(cfg, training, model_meta=None, **kwargs):
        """Configure for Task Adaptation Task
        """
        task_adapt_type = cfg['task_adapt'].get('type', None)
        adapt_type = cfg['task_adapt'].get('op', 'REPLACE')

        model_tasks, dst_classes = None, None
        model_classes, data_classes = [], []
        train_data_cfg = Stage.get_data_cfg(cfg, "train")
        if isinstance(train_data_cfg, list):
            train_data_cfg = train_data_cfg[0]