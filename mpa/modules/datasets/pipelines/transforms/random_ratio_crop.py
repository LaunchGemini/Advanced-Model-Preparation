# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numbers
import random

import numpy as np
import mmcv

from mmcls.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomRatioCrop(object):
    def __init__(self,
                 crop_ratio,
                 padding_ratio=None,
                 pad_if_needed=False,
                 pad_val=0,
                 padding_mode='constant'):
        if isinstance(crop_ratio, (tuple, list)):
            self.ratio = crop_ratio
        else:
            self.ratio = (crop_ratio, crop_ratio)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding_ratio
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, ratio):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            ratio (tuple): Expected output size ratio of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        height_ratio, width_ratio = ratio
        if width_ratio == 1.0 and height_ratio == 1.0:
            return 0, 0, height, width

        target_height = int(height * height_ratio)
        target_width = int(width * width_ratio)
        ymin = random.randint(0, height - target_height)
        xmin = random.randint(0, width - target_width)
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.padding is not None:
                if isinstance(self.padding, numbers.Number):
                    padding = int(img.shape[0] * self.padding)
                elif isinstance(self.padding, tuple) and len(self.padding) in [2, 4]:
                    padding = (
                        int(img.shape[0] * self.padding[0]),
                        int(img.shape[1] * self.padding[1]),
                        int(img.shape[0] * self.padding[0]),
                        int(img.shape[1] * self.padding[1])
                    )
                else:
                    ValueError('padding_ratio should a float or 2 or 4 elements tuple but '
                               f'recived {self.padding}')
                img = mmcv.impad(
                    img, padding=padding, pad_val=self.pad_val)

            # pad the height if needed
            # if self.pad_if_needed and img.shape[0] < self.size[0]:
                # img