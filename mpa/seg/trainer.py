# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import time
import numbers
import glob
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import mmcv
from mmcv import get_git_hash

from mmseg import __version__
# from mmseg.apis import train_segmentor
from .train import train_segmentor
# from m