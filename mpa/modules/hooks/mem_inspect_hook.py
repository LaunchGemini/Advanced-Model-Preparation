# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import torch
import pandas as pd

from mmcv.runner import Hook, HOOKS

from .utils import plot_mem, print_report
from mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class MemInspectHook(Hook):
    def __init__(self, **kwargs):
        super(MemInspectHook, self).__init__()
        self.exp = kwargs.get('exp', 'baseline')
        self.print_report = kwargs.get('print_report', False)
        self.output_file = kwargs.get('output_file', f'gpu_mem_plot_{self.exp}.png')
        data_cfg = kwargs.get('data_cfg', None)
        if data_cfg is None:
            raise ValueError('cannot find data config')
        logger.info(f'data_cfg = {data_cfg}')

        self.data_args = self._parse_data_cfg(data_cfg)
        logger.info(f'keys in data args = {self.data_args.keys()}')

        # input value will be passed as positional argument to the model
        self.input = self.data_args.pop('input', None)
        if self.input is None:
            raise ValueError("invalid data configuration. 'input' key is the mandatory for the data configuration.")

    def _parse_data_cfg(self, cfg):
        input_args = {}
     