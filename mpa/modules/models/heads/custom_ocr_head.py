# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import force_fp32

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.losses import accuracy
from mmseg.models.decode_heads.ocr_head import OCRHead
from mmseg.core import add_prefix
from mpa.modules.utils.seg_utils import get_valid_label_mask_per_batch


@HEADS.register_module()
class CustomOCRHead(OCRHead):
    """Custom Object-Contextual Representations for Semantic Segmentation.
    """

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg, pixel_weights=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
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
        seg_logits = self.forward(inputs, prev_output)
        valid_label_mask = get_valid_label_mask_per_batch(img_metas, self.num_classes)
        losses = self.losses(seg_logits, gt_semantic_seg, valid_label_mask, train_cfg, pixel_weights)

        return losses, seg_