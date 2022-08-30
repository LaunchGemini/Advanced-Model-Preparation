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
    n = len(gallery)
    x = np.reshape(query, (m, -1))
    y = np.reshape(gallery, (n, -1))
    if metric == 'euclidean':
        dist = euclidean_distances(x, y)
    elif metric == 'cosine':
        dist = cosine_distances(x, y)
    else:
        raise KeyError("Unsupported distance metric:", metric)

    return dist


def mean_ap(distmat, query_ids, gallery_ids):
    distmat = distmat
    m, n = distmat.shape
    if gallery_ids is None:
        gallery_ids = query_ids
        cut = True
    else:
        cut = False

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    if cut:
        indices = indices[:, 1:]
        matches = matches[:, 1:]
    # Compute AP for each query
    aps = []
    for i in range(m):
        y_true = matches[i]
        y_score = -distmat[i][indices[i]]
        if not np.any(y_true):
            if y_score[0] < -0.5:
                aps.append(1)
            else:
                aps.append(0)
        else:
            aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        # raise RuntimeError("No valid query")
        return 0
    return np.mean(aps)


def calculate_cmc(distmat, query_ids, gallery_ids, topk=100, first_match_break=True):
    if gallery_ids is None:
        gallery_ids = query_ids
    distmat = distmat.cpu().numpy()
    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        if not np.any(matches[i]):
            continue
        repeat = 1
        for _ in range(repeat):
            index = np.nonzero(matches[i])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        # raise RuntimeError("No valid query")
        return [0]
    return ret.cumsum() / num_valid_queries


def init_dist(args, backend="nccl"):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    if not dist.is_available():
        args.launcher = "none"

    if args.launcher == "pytorch":
        # DDP
        init_dist_pytorch(args, backend)
        return True

    elif args.launcher == "slurm":
        # DDP
        init_dist_slurm(args, backend)
        return True

    elif args.launcher == "none":
        # DataParallel or single GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            args.total_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            args.total_gpus = torch.cuda.device_count()
        if args.total_gpus > 1:
            warnings.warn(
                "It is highly recommended to use DistributedDataParallel by setting "
                "args.launcher as 'slurm' or 'pytorch'."
            )
        return False

    else:
        