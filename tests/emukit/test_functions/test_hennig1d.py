# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions.quadrature_functions import hennig1D


def test_hennig1D_return_shape():
    """
    Test output dimension is 2d
    """
    _, hennig1d_func = hennig1D()
    x = np.zeros((2, 1))
    result = hennig1d_func(x)
    assert result.ndim == 2
    assert result.shape == (2, 1)
