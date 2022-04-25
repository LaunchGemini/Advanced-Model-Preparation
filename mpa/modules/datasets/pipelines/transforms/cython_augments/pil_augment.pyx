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
    cdef ImageInfo info
    cdef unsigned long long ptr_val

    info.width = image.size[0]
    info.height = image.size[1]

    ptr_val = dict(image.getdata().unsafe_ptrs)['image']
    info.img_ptr = (<PixelRGBA**>ptr_val)

    return info


cdef inline int L24(PixelRGBA rgb):
    return rgb.r * 19595 + rgb.g * 38470 + rgb.b * 7471 + 0x8000


cdef inline unsigned char clip(float v):
    if v < 0.0:
        return 0
    if v >= 255.0:
        return 255

    return <unsigned char>v


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_lut(image: Image, int[:] lut):
    cdef ImageInfo info
    info = parse_img_info(image)

    for y in range(info.height):
        for x in range(info.width):
            info.img_ptr[y][x].r = lut[info.img_ptr[y][x].r]
            info.img_ptr[y][x].g = lut[info.img_ptr[y][x].g + 256]
            info.img_ptr[y][x].b = lut[info.img_ptr[y][x].b + 512]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:] c_histogram(image: Image):
    cdef ImageInfo info
    cdef int x, y
    cdef int[:] hist = cvarray(shape=(768,), itemsize=sizeof(int), format="i")

    info = parse_img_info(image)

    for x in range(768):
        hist[x] = 0

    for y in range(info.height):
        for x in range(info.width):
            hist[info.img_ptr[y][x].r] += 1
            hist[info.img_ptr[y][x].g + 256] += 1
            hist[info.img_ptr[y][x].b + 512] += 1

    return hist


def histogram(image: Image):
    cdef int[:] hist = c_histogram(image)
    cdef int i
    cdef int return_vals[768]

    for i in range(768):
        return_vals[i] = hist[i]

    return return_vals


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def autocontrast(image: Image, cutoff=0, ignore=None):
    if image.mode != "RGB":
        image = image.convert("RGB")
    cdef int layer = 0
    cdef int* h
    cdef int i, lo, hi, ix, cut, n
    cdef double scale, offset
    cdef int[:] histogram
    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")

    histogram = c_histogram(image)

    for layer in range(0, 768, 256):
        h = &histogram[layer]

        if ignore is not None:
            # get rid of outliers
            try:
                h[ignore] = 0
            except TypeError:
                # assume sequence
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            # cut off pixels from both ends of the histogram
            if not isinstance(cutoff, tuple):
                cutoff = (cutoff, cutoff)
            # get number of pixels
            n = 0
            for ix in range(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = n * cutoff[0] // 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the high end
            cut = n * cutoff[1] // 100
            for hi in range(255, -1, -1):
       