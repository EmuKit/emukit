# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions import forrester_function
from emukit.test_functions.forrester import forrester_low


def test_forrester_return_shape():
    """
    Test output dimension is 2d
    """
    branin, _ = forrester_function()
    x = np.array([[0.75], [0.10]])
    assert branin(x).ndim == 2
    assert branin(x).shape == (2, 1)


def test_forrester_low_return_shape():
    x = np.array([[0.2], [0.4]])
    assert forrester_low(x, 0).shape == (2, 1)
