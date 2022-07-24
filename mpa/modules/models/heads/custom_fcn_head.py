# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import force_fp32

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.losses import accuracy
from mmseg.ops import resize
from mmseg.core import add_prefix
from mpa.modules.utils.seg_utils import get_valid_label_mask_per_batch


@HEADS.register_module()
class CustomFCNHead(FCNHead):
    """Custom Fully Convolution Networks for Semantic Segmentation.
    """

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, pixel_weights=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', 'img_norm_cfg',
                and 'ignored_labels'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        valid_label_mask = get_valid_label_mask_per_batch(img_metas, self.num_classes)
        losses = self.losses(seg_logits, gt_semantic_seg, valid_label_mask, train_cfg, pixel_weights)

        if self.forward_output is not None:
            return losses, self.forward_output
        else:
            return losses, seg_logits

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, valid_label_mask, train_cfg, pixel_weights=None):
        """Compute segmentation loss."""

        loss = dict()

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        seg_label = seg_label.squeeze(1)
        out_losses = dict()
        for loss_idx, loss_module in enumerate(self.loss_modules):
            loss_value, loss_meta = loss_module(
                seg_logit,
                seg_label,
                valid_label_mask,
                pixel_weights=pixel_weights
          