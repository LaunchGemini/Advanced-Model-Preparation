
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES
from mmdet.models.losses.focal_loss import sigmoid_focal_loss, py_sigmoid_focal_loss
from mmdet.models.losses.varifocal_loss import varifocal_loss


def cross_sigmoid_focal_loss(inputs,
                             targets,
                             weight=None,
                             num_classes=None,
                             alpha=0.25,
                             gamma=2,
                             reduction="mean",
                             avg_factor=None,
                             use_vfl=False,
                             valid_label_mask=None):
    """
    Arguments:
       - inputs: inputs Tensor (N * C)
       - targets: targets Tensor (N), if use_vfl, then Tensor (N * C)
       - weights: weights Tensor (N), consists of (binarized label schema * weights)
       - num_classes: number of classes for training
       - alpha: focal loss alpha
       - gamma: focal loss gamma
       - reduction: default = mean
       - avg_factor: average factors
    """
    cross_mask = inputs.new_ones(inputs.shape, dtype=torch.int8)
    if valid_label_mask is not None:
        neg_mask = targets.sum(axis=1) == 0 if use_vfl else targets == num_classes
        neg_idx = neg_mask.nonzero(as_tuple=True)[0]
        cross_mask[neg_idx] = valid_label_mask[neg_idx].type(torch.int8)

    if use_vfl:
        calculate_loss_func = varifocal_loss
    else:
        if torch.cuda.is_available() and inputs.is_cuda:
            calculate_loss_func = sigmoid_focal_loss
        else:
            inputs_size = inputs.size(1)
            targets = F.one_hot(targets, num_classes=inputs_size+1)
            targets = targets[:, :inputs_size]
            calculate_loss_func = py_sigmoid_focal_loss

    loss = calculate_loss_func(inputs,
                               targets,
                               weight=weight,
                               gamma=gamma,
                               alpha=alpha,
                               reduction='none',
                               avg_factor=None) * cross_mask

    if reduction == "mean":
        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss
