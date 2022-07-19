# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
from mmcv.runner import force_fp32

from mmdet.core import bbox_overlaps, reduce_mean, multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.atss_head import ATSSHead
from mpa.modules.models.heads.cross_dataset_detector_head import \
    CrossDatasetDetectorHead
from mpa.modules.models.losses.cross_focal_loss import CrossSigmoidFocalLoss

EPS = 1e-12


@HEADS.register_module()
class CustomATSSHead(CrossDatasetDetectorHead, ATSSHead):
    def __init__(
        self,
        *args,
        bg_loss_weight=-1.0,
        use_qfl=False,
        qfl_cfg=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        **kwargs
    ):
        if use_qfl:
            kwargs['loss_cls'] = qfl_cfg
        super().__init__(*args, **kwargs)
        self.bg_loss_weight = bg_loss_weight
        self.use_qfl = use_qfl

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, 