# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions.quadrature import sombrero2D


def test_sombrero2D_return_shape():
    """
    Test output dimension is 2d
    """
    sombrero2D_func, _ = sombrero2D()
    x = np.ones((3, 2))
    result = sombrero2D_func.f(x)
    assert result.ndim == 2
    assert result.shape == (3, 1)
