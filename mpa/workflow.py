# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# from datetime import datetime as dt

from mpa.stage import Stage
from mpa.utils.config_utils import copy_config


class Workflow(object):
    def __init__(self, stages, workflow_hooks=None):
        if not isinstance(stages, list):
            raise ValueError('stages parameter should be the list of Stage instance')
        if len(stages) == 0:
            raise ValueError('required one or more stage for the workflow')
        for stage in stages:
            if not isinstance(stage, Stage):
                raise ValueError('stages parameter should be the list of Stage instance')
        if workflow_hooks is not None and not isinstance(workflow_hooks, list):
            raise ValueError('workflow_hooks should be a list')

        self.stages = stages
        self.workflow_hooks = workflow_hooks
        self.results = {}
        self.context = {stage.name: {} for stage in stages}

    def _call_wf_hooks(self, fname, stage_idx=-1):
        if self.workflow_hooks is not None:
            for hook in self.workflow_hooks:
                getattr(hook, fname)(self, stage_idx, self.results)

    def run(self, **kwargs):
        model_cfg = kwargs.get('model_cfg', None)
        data_cfg = kwargs.get('data_cfg', None)
        model_ckpt = kwargs.get('model_ckpt', None)
        # output_path = kwargs.get('output_path', '.')
        mode = kwargs.get('mode', 'train')
        ir_path = kwargs.get('ir_path', None)

        self._call_wf_hooks('before_workflow')
        for i, stage in enumerate(self.stages):
            self._call_wf_hooks('before_stage', i)

            # create keyword arguments that will be passed to stage.run() refer to input map defined in config
            stage_kwargs = dict()
            if hasattr(stage, 'input'):
                for arg_name, arg in stage.input.items():
                    stage_name = arg.get('stage_name', None)
                    output_key = arg.get('output_key', None)
                    if stage_name is None or output_key is None:
                        raise ValueError(f"'stage_name' and 'output_key' attributes are required for the '{arg_name}'")
                    stage_kwargs[arg_name] = self.context[stage_name].get(output_key, None)

            # context will keep the results(path to model, etc) of each stage
            # stage.run() returns 