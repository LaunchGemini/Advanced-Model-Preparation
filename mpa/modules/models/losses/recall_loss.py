# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight
from mmseg.models.losses.pixel_base import BasePixelLoss


def recallCE(input,
             target,
             class_weight=None,
             ignore_index=255):

    _, c, _, _ = input.size()

    pred = input.argmax(dim=1)
    idex = (pred != target).view(-1)

    # recall loss
    gt_counter = torch.ones((c)).to(target.device)
    gt_idx, gt_count = torch.unique(target,