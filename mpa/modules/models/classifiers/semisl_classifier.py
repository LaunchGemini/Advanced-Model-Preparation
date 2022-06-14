# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn

from mmcls.models.builder import CLASSIFIERS, build_backbone, build_neck, build_head

from mpa.modules.models.classifiers.sam_classifier import SAMClassifier
from mpa.utils.logger import get_logger

logger = get_logger()


@CLASSIFIERS.register_module()
class SemiSLClassifier(SAMClassifier):
    """ Semi-SL Classifier

    The classifier is a classifier that supports Semi-SL task
    that handles unlabeled data.

    Args:
        backbone (dict): backbone network configuration
        neck (dict): model neck configuration
        head (dict): model head configuration
        pretrained (str or boolean): Initialize to pre-trained weight
            according to the backbone when the path
            or boolean True of pre-trained weight is performed.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 hea