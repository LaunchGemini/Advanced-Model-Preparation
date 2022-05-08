
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.datasets import PIPELINES, DATASETS, build_dataset
# import torch
import numpy as np

from mpa.modules.utils.task_adapt import map_class_names, map_cat_and_cls_as_order


@DATASETS.register_module()
class TaskAdaptEvalDataset(object):
    """Dataset wrapper for task-adative evaluation.
    """
    def __init__(self, model_classes, **kwargs):
        dataset_cfg = kwargs.copy()
        org_type = dataset_cfg.pop('org_type')
        dataset_cfg['type'] = org_type
        self.dataset = build_dataset(dataset_cfg)
        self.model_classes = model_classes
        self.CLASSES = self.dataset.CLASSES
        self.data2model = map_class_names(self.CLASSES, self.model_classes)
        if org_type == 'CocoDataset':
            self.dataset.cat2label, self.dataset.cat_ids = map_cat_and_cls_as_order(
                self.CLASSES, self.dataset.coco.cats)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def evaluate(self, results, **kwargs):
        # Filter & reorder detection results
        adapt_results = []
        for result in results:  # for each image
            adapt_result = []
            for model_class_index in self.data2model:  # for each class
                # Gather per-class results according to index mapping
                if model_class_index >= 0:
                    adapt_result.append(result[model_class_index])
                else:
                    adapt_result.append(np.empty([0, 5]))
            adapt_results.append(adapt_result)

        # Call evaluation w/ org arguments