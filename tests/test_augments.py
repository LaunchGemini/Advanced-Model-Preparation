# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import pytest
import numpy as np
from PIL import Image
from typing import Any, List, Union
from mpa.modules.datasets.pipelines.transforms.augments import Augments, CythonAugments
from copy import deepcopy
import cv2


@pytest.fixture
def images() -> List[Image.Image]:
    n_seed = 3003
    n_imgs = 4
    n_shapes = 4
    img_size = 50
    size = [img_size, img_size, 3]

    np.random.seed(n_seed)

    imgs = []
    for _ in range(n_imgs):
        img = np.full(size, 0, dtype=np.uint8)
        for _ in range(n_shapes):
            position = np.random.randint(0, 50, size=[2]).tolist()
            color = np.random.randint(0, 256, size=[3]).tolist()
            marker_type = np.random.randint(0, 7)
            img = cv2.drawMarker(img, position, color, marker_type, thickness=5)
        imgs += [Image.fromarray(img)]

    return imgs


EXACT_EQUAL_TESTS = [
    (