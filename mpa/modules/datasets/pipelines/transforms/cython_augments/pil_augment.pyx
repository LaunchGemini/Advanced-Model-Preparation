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
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        # find lowest/highest samples after preprocessing
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            for i in range(256):
                lut[layer + i] = i
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                i = ix
                ix = (int)(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut[layer + i] = ix

        layer += 256

    _c_lut(image, lut)

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def equalize(image: Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    cdef int[:] h
    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")

    h = c_histogram(image)

    cdef int b, histo_len, histo_sum, i, n, step, num
    cdef int histo[256]

    for b in range(0, 768, 256):
        histo_len = 0
        histo_sum = 0

        for i in range(256):
            num = h[b + i]
            if num > 0:
                histo[histo_len] = num
                histo_sum += num
                histo_len += 1

        if histo_len <= 1:
            for i in range(256):
                lut[b + i] = i
        else:
            step = (histo_sum - histo[histo_len - 1]) // 255
            if not step:
                for i in range(256):
                    lut[b + i] = i
            else:
                n = step // 2
                for i in range(256):
                    lut[b + i] = n // step
                    n = n + h[i + b]

    _c_lut(image, lut)

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def posterize(image: Image, bits: int):
    if image.mode != "RGB":
        image = image.convert("RGB")

    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")
    cdef int i, b, c_bits
    cdef unsigned char mask

    c_bits = bits

    mask = ~(2 ** (8 - c_bits) - 1)
    for b in range(0, 768, 256):
        for i in range(256):
            lut[b + i] = i & mask
    _c_lut(image, lut)
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solarize(image: Image, threshold: int = 128):
    if image.mode != "RGB":
        image = image.convert("RGB")

    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")
    cdef int i, b, c_threshold
    cdef ImageInfo info

    c_threshold = threshold

    for b in range(0, 768, 256):
        for i in range(256):
            if i < c_threshold:
                lut[b + i] = i
            else:
                lut[b + i] = 255 - i

    _c_lut(image, lut)
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
def color(im