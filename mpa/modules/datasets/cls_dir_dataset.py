# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import copy
import numpy as np

from mmcv.utils.registry import build_from_cfg
from mmcls.datasets.builder import DATASETS, PIPELINES
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.pipelines import Compose

from mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class ClsDirDataset(BaseDataset):
    """Classification dataset for Finetune, Incremental Learning, Semi-SL
        assumes the data for classification is divided by folders (classes)

    Args:
        data_dir (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        classes (list): List of classes to be used for training,
            If empty, use all classes in the folder list
            Also, if there is new_classes (incremental learning),
            classes are used as a list of old classes
        new_classes (list): List of final classes to be used for incremental learning
        use_labels (bool): dataset with labels or unlabels
    """

    def __init__(self, data_dir, pipeline=[], classes=[], new_classes=[], use_labels=True, **kwargs):
        self.data_dir = data_dir
        self._samples_per_gpu = kwargs.pop('samples_per_gpu', 1)
        self._workers_per_gpu = kwargs.pop('workers_per_gpu', 1)
        self.use_labels = use_labels
        self.img_indices = dict(old=[], new=[])
        self.class_acc = False

        self.new_classes = new_classes
        if not classes:
            self.CLASSES = self.get_classes_from_dir(self.data_dir)
        else:
            self.CLASSES = self.get_classes(classes)
        if isinstance(self.CLASSES, list):
            self.CLASSES.sort()
        self.num_classes = len(self.CLASSES)

        # Pipeline Settings
        if isinstance(pipeline, dict):
            self.pipeline = {}
            for k, v in pipeline.items():
                _pipeline = [dict(type='LoadImageFromFile'), *v]
                _pipeline = [build_from_cfg(p, PIPELINES) for p in _pipeline]
                self.pipeline[k] = Compose(_pipeline)
            self.num_pipes = len(pipeline)
        elif isinstance(pipeline, list):
            self.num_pipes = 1
            _pipeline = [dict(type='LoadImageFromFile'), *pipeline]
            self.pipeline = Compose([build_from_cfg(p, PIPELINES) for p in _pipeline])

        self.data_infos = self.load_annotations()
        self.statistics()

    def statistics(self):
        logger.info(f'ClsDirDataset - {self.num_classes} classes from {self.data_dir}')
        logger.info(f'- Classes: {self.CLASSES}')
        if self.new_classes:
            logger.info(f'- New Classes: {self.new_classes}')
            old_data_length = len(self.img_indices['old'])
            new_data_length = len(self.img_indices['new'])
            logger.info(f'- # of old classes images: {old_data_length}')
            logger.info(f'- # of New classes images: {new_data_length}')
        logger.info(f'- # of imag