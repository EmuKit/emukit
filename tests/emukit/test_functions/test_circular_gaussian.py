# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions.quadrature import circular_gaussian


def test_circular_gaussian_return_shape():
    """
    Test output dimension is 2d
    """
    circular_gaussian_func, _ = circular_gaussian()
    x = np.ones((3, 2))
    result = circular_gaussian_func.f(x)
    assert result.ndim == 2
    assert result.shape == (3, 1)
