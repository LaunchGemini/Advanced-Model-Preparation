# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from mmcls.datasets.builder import DATASETS
from mmcls.models.losses import accuracy
from mmcls.core.evaluation import f1_score, precision, recall

from .cls_csv_dataset import CSVDatasetCls


@DATASETS.register_module()
class MultiClsDataset(CSVDatasetCls):
    def __init__(self, tasks=None, **kwargs):
        self.tasks = tasks
        super(MultiClsDataset, self).__init__(**kwargs)

    def load_annotations(self):
        dataframe = self._read_csvs()
        data_infos = []
        for _, data in dataframe.iterrows():
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': data['ImagePath']}
            gt_labels = []
            for task, cls in zip(self.tasks.keys(), self.tasks.values()):
                gt_labels += [cls.index(data[task])]
            info['gt_label'] = np.array(gt_labels, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options={'topk': (1, )},
                 logger=None):
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['accuracy', 'precision', 'recall', 'f1_score',
                           'class_accuracy']
        eval_results = {}
        if results:
            results = self._refine_results(results)
            for metric in metrics:
                if metric not in allowed_metrics:
                    raise KeyError(f'metric {metric} is not supported.')
                gt_labels = self.get_gt_labels()
                gt_labels = np.transpose(gt_labels)
                for task, gt in zip(self.tasks.keys(), gt_labels):
                    res = results[task]
                    num_imgs = len(res)
                  