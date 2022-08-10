# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from torch.optim.optimizer import Optimizer, required

from mmcv.runner import OPTIMIZERS


@OPTIMIZERS.register_module(