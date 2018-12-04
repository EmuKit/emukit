# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions.quadrature_functions import circular_gaussian


def test_circular_gaussian_return_shape():
    """
    Test output dimension is 2d
    """
    _, circular_gaussian_func = circular_gaussian()
    x = np.ones((3, 2))
    assert circular_gaussian_func(x).ndim == 2
    assert circular_gaussian_func(x).shape == (3, 1)
