# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import numbers
import os.path as osp
import time
import warnings
import torch
import numpy as np
import random

import torch.multiprocessing as mp
import torch.distributed as dist

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Fp16OptimizerHook, build_optimizer, build_runner, HOOKS

from mmcls import __version__
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier
from mmcls.utils import collect_env
from mmcls.core import DistOptimizerHook

from mpa.registry import STAGES
from mpa.modules.datasets.composed_dataloader import ComposedDL
from mpa.stage import Stage
from mpa.cls.stage import ClsStage
from mpa.modules.hooks.eval_hook import CustomEvalHook, DistCustomEvalHook
from mpa.modules.hooks.fp16_sam_optimizer_hook import Fp16SAMOptimizerHook
from mpa.utils.logger import get_logger
from mpa.utils.data_cpu import MMDataCPU

logger = get_logger()


@STAGES.register_module()
class ClsTrainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=True, **kwargs)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        distributed = False
        if cfg.gpu_ids is not None:
            if isinstance(cfg.get('gpu_ids'), numbers.Number):
                cfg.gpu_ids = [cfg.get('gpu_ids')]
            if len(cfg.gpu_ids) > 1:
                distributed = True

        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)

        # Data
        if 'unlabeled' in cfg.data:
            datasets = [[build_dataset(cfg.data.train), build_dataset(cfg.data.unlabeled)]]
        else:
            datasets = [build_dataset(cfg.data.train)]

        # Dataset for HPO
        hp_config = kwargs.get('hp_config', None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDataset(datasets[0], hp_config)

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed

        if isinstance(datasets[0], list):
            repr_ds = datasets[0][0]
        else:
            repr_ds = datasets[0]

        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmcls_version=__version__)
            if hasattr(repr_ds, 'tasks'):
                cfg.checkpoint_config.meta['tasks'] = repr_ds.tasks
            else:
                cfg.checkpoint_config.meta['CLASSES'] = repr_ds.CLASSES
            if 'task_adapt' in cfg:
                if hasattr(self, 'model_tasks'):  # for incremnetal learning
                    cfg.checkpoint_config.meta.update({'tasks': self.model_tasks})
                    # instead of update(self.old_tasks), update using "self.model_tasks"
                else:
                    cfg.checkpoint_config.meta.update({'CLASSES': self.model_classes})

        if distributed:
            if cfg.dist_params.get('linear_scale_lr', False):
                new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
                logger.info(f'enabled linear scaling rule to the learning rate. \
                    changed LR from {cfg.optimizer.lr} to {new_lr}')
                cfg.optimizer.lr = new_lr

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.yaml')) # FIXME bug to save
        # logger.info(f'Config:\n{cfg.pretty_text}')

        if distributed:
            os.environ['MASTER_ADDR'] = cfg.dist_params.get('master_addr', 'localhost')
            os.environ['MASTER_PORT'] = cfg.dist_params.get('master_port', '29500')

            mp.spawn(ClsTrainer.train_worker, nprocs=len(cfg.gpu_ids),
                     args=(datasets, cfg, distributed, True, timestamp, meta))
        else:
            ClsTrainer.train_worker(None, datasets, cfg,
                                    distributed,
                                    True,
                                    timestamp,
                                    meta)

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, 'best_model.pth'
                                    if osp.exists(osp.join(cfg.work_dir, 'best_model.pth'))
                                    else 'latest.pth')
        return dict(final_ckpt=output_ckpt_path)

    @staticmethod
    def train_worker(gpu, dataset, cfg, distributed, validate, timestamp, meta):
        logger.info(f'called train_worker() gpu={gpu}, distributed={distributed}, validate={validate}')
        if distributed:
            torch.cuda.set_device(gpu)
            dist.init_process_group(backend=cfg.dist_params.get('backend', 'nccl'),
                                    world_size=len(cfg.gpu_ids), rank=gpu)
            logger.info(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

        # model
        model = build_classifier(cfg.model)

        # prepare data loaders
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        train_data_cfg = Stage.get_data_cfg(cfg, "train")
        ote_dataset = train_data_cfg.get('ote_dataset', None)
        drop_last = False
        dataset_len = len(ote_dataset) if ote_dataset else 0
        # if task == h-label & dataset size is bigger than batch size
        if train_data_cfg.get('hierarchical_info', None) and dataset_len > cfg.data.get('samples_per_gpu', 2):
            drop_last = True
        # updated to adapt list of dataset for the 'train'
        data_loaders = []
        sub_loaders = []
        for ds in dataset:
            if isinstance(ds, list):
                sub_loaders = [
                    build_dataloader(
                        sub_ds,
                        sub_ds.samples_per_gpu if hasattr(sub_ds, 'samples_per_gpu') else cfg.data.samples_per_gpu,
                        sub_ds.workers_per_gpu if hasattr(sub_ds, 'workers_per_gpu') else cfg.data.workers_per_gpu,
                        num_gpus=len(cfg.gpu_ids),
                        dist=distributed,
                        round_up=True,
                        seed=cfg.seed,
                        drop_last=drop_last,
                        persistent_workers=False
                    ) for sub_ds in ds
                ]
                data_loaders.append(ComposedDL(sub_loaders))
            else:
                data_loaders.append(
                    build_dataloader(
                        ds,
                        cfg.data.samples_per_gpu,
                        cfg.data.workers_per_gpu,
                        # cfg.gpus will be ignored if distributed
                        num_gpus=len(cfg.gpu_ids),
                        dist=distributed,
                        round_up=True,
                        seed=cfg.seed,
                        drop_last=drop_last,
                        persistent_workers=False
                    ))

        # put model on gpus
        if torch.cuda.is_available():
            model = model.cuda()

        # put model on gpus
        if torch.cuda.is_available():
            if distributed:
                # put model on gpus
                find_unused_parameters = cfg.get('find_unused_parameters', False)
                # Sets the `find_unused_parameters` parameter in
                # torch.nn.parallel.DistributedDataParallel
                model = MMDistributedDataParallel(
                    model,
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                model = MMDataParallel(
                    model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        else:
            model = MMDataCPU(model)

        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)

        if cfg.get('runner') is None:
            cfg.runner = {
                'type': 'EpochBasedRunner',
                'max_epochs': cfg.total_epochs
            }
            warnings.warn(
                'config is now expected to have a `runner` section, '
                'please set `runner` in your config.', UserWarning)

        runner = build_runner(
            cfg.runner,
            default_args=dict(
