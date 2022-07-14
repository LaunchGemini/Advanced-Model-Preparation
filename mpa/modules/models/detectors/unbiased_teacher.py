# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import numpy as np
import copy
import functools
from collections import OrderedDict

from mmdet.models import DETECTORS, build_detector
from mmdet.models.detectors import BaseDetector
from .sam_detector_mixin import SAMDetectorMixin
from mpa.utils.logger import get_logger

logger = get_logger()


@DETECTORS.register_module()
class UnbiasedTeacher(SAMDetectorMixin, BaseDetector):
    """Unbiased teacher frameowork for general detectors
    """

    def __init__(
        self,
        unlabeled_loss_weight=1.0,
        unlabeled_loss_names=['loss_cls', ],
        pseudo_conf_thresh=0.7,
        enable_unlabeled_loss=False,
        bg_loss_weight=-1.0,
        **kwargs
    ):
        super().__init__()
        self.unlabeled_loss_weight = unlabeled_loss_weight
        self.unlabeled_loss_names = unlabeled_loss_names
        self.pseudo_conf_thresh = pseudo_conf_thresh
        self.unlabeled_loss_enabled = enable_unlabeled_loss
        self.bg_loss_weight = bg_loss_weight

        cfg = kwargs.copy()
        arch_type = cfg.pop('arch_type')
        cfg['ty