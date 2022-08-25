# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import subprocess
import warnings

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics import average_precision_score
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch import nn


def distance_metric(query, gallery, metric='euclidean'):
    if gallery is None:
        gallery = query
    m = len(query)
    n =