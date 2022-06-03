# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import math
from mmcv.runner import HOOKS, Hook
from mmcv.runner.hooks.ema import EMAHook
from mmcv.parallel import is_module_wrapper
from mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class DualModelEMAHook(Hook):
    """Generalized re-implementation of mmcv.runner.EMAHook

    Source model paramters would be exponentially averaged
    onto destination model pararmeters on given intervals

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        epoch_momentum (float): if > 0, momentum is ignored and re-calculated.
            momentum = 1 - exp(1 - epoch_momentum, 1/num_iter_per_epoch)
        