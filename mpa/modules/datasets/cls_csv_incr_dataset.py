
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.datasets.builder import DATASETS
from .multi_cls_dataset import MultiClsDataset
from .cls_csv_dataset import CSVDatasetCls
from mpa.modules.utils.task_adapt import map_class_names
import numpy as np


@DATASETS.register_module()
class LwfTaskIncDataset(MultiClsDataset):
    def __init__(self, pre_stage_res=None, model_tasks=None, **kwargs):
        self.pre_stage_res = pre_stage_res
        self.model_tasks = model_tasks
        if self.pre_stage_res is not None:
            self.pre_stage_data = np.load(self.pre_stage_res, allow_pickle=True)
            for p in kwargs['pipeline']:
                if p['type'] == 'Collect':
                    p['keys'] += ['soft_label']
        super(LwfTaskIncDataset, self).__init__(**kwargs)
