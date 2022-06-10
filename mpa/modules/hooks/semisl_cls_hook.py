# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

from mmcv.runner import Hook, HOOKS
from mmcv.parallel import is_module_wrapper


@HOOKS.register_module()
class SemiSLClsHook(Hook):
    """Hook for SemiSL for classification

    This hook includes unlabeled warm-up loss coefficient (default: True):
        unlabeled_coef = (0.5 - cos(min(pi, 2 * pi * k) / K)) / 2
        k: current step, K: total steps
    Also, this hook adds semi-sl-related data to the log (unlabeled_loss, pseudo_label)

    Args:
        total_steps (int): total steps for training (iteration)
            Raise the coefficient from 0 to 1 during half the duration of total_steps
            default: 0, use runner.max_iters
        unlabeled_warmup (boolean): enable unlabeled warm-up loss coefficient
            If False, Semi-SL uses 1 as unlabeled loss coefficient
    """

    def __init__(self,
                 total_steps=0,
                 unlabeled_warmup=True,
                