# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.mask_rcnn import MaskRCNN
from .sam_detector_mixin import SAMDetectorMixin
from .l2sp_detector_mixin import L2SPDetectorMixin
from mpa.modules.utils.task_adapt import map_class_names
from mpa.utils.logger import get_logger

logger = get_logger()


@DETECTORS.register_module()
class CustomM