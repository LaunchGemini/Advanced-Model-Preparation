# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import cython
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
import cv2
from PIL import Image
from PIL.Image import Resampling
np.import_array()


cdef struct PixelRGBA:
    unsigned char r
    unsigned char g
    unsigned char b
    unsigned char a


cdef struct ImageInfo:
    int width
    int height
    PixelRGBA** img_ptr


cdef ImageInfo parse_img_info(image: Image):
    cdef I