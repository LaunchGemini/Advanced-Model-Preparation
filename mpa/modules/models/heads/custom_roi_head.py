# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.integration.nncf.utils import no_nncf_trace
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mpa.modules.models.heads.cross_dataset_detector_head import (
    CrossDatasetDetectorHead,
)
from mpa.modules.models.losses.cross_focal_loss import CrossSigmoidFocalLoss


@HEADS.register_module()
class CustomRoIHead(StandardRoIHead):
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        if bbox_head.type == 'Shared2FCBBoxHead':
            bbox_head.type = 'CustomConvFCBBoxHead'
        self.bbox_head = build_head(bbox_head)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        labels, label_weights, bbox_targets, bbox_weights, valid_label_mask = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, img_metas, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                