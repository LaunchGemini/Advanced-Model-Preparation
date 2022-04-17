# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Union
from PIL import Image, ImageOps, ImageEnhance
from PIL.Image import Resampling
import mpa.modules.datasets.pipelines.transforms.cython_augments.pil_augment as pil_aug
import numpy as np
import random

PILImage = Image.Image
CvImage = np.ndarray
ImgTypes = Union[PILImage, CvImage]


class Augments:
    def _check_args_tf(kwargs):
        def _interpolation(kwargs):
            interpolation = kwargs.pop("resample", Resampling.BILINEAR)
            if isinstance(interpolation, (list, tuple)):
                return random.choice(interpolation)
            else:
                return interpolation

        kwargs["resample"] = _interpolation(kwargs)

    @staticmethod
    def autocontrast(img: PILImage, *args, **kwargs) -> PILImage:
        return ImageOps.autocontrast(img)

    @staticmethod
    def equalize(img: PILImage, *args, **kwargs) -> PILIma