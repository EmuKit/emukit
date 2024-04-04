# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions import branin_function


def test_branin_minima():
    """
    Test some known values at minima
    """
    branin, _ = branin_function()
    minimum = 0.39788  # value of function at minima
    assert np.isclose(branin(np.array([[-np.pi, 12.275]])), minimum, atol=1e-3)
    assert np.isclose(branin(np.array([[np.pi, 2.275]])), minimum, atol=1e-3)
    assert np.isclose(branin(np.array([[9.42478, 2.475]])), minimum, atol=1e-3)


def test_branin_return_shape():
    """
    Test output dimension is 2d
    """
    branin, _ = branin_function()
    x = np.array([[7.5, 7.5], [10.0, 10.0]])
    assert branin(x).ndim == 2
    assert branin(x).shape == (2, 1)
