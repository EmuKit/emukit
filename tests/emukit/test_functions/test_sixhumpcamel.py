# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions import sixhumpcamel_function


def test_sixhumpcamel_minima():
    """
    Test some known values at minima
    """
    sixhumpcamel, _ = sixhumpcamel_function()
    minimum = -1.0316  # value of function at minima
    assert np.isclose(sixhumpcamel(np.array([[0.0898, -0.7126]])), minimum, atol=1e-3)
    assert np.isclose(sixhumpcamel(np.array([[-0.0898, 0.7126]])), minimum, atol=1e-3)


def test_sixhumpcamel_return_shape():
    """
    Test output dimension is 2d
    """
    sixhumpcamel, _ = sixhumpcamel_function()
    x = np.array([[7.5, 7.5], [10.0, 10.0]])
    assert sixhumpcamel(x).ndim == 2
    assert sixhumpcamel(x).shape == (2, 1)
