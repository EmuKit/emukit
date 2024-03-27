# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.core.acquisition import Acquisition


@pytest.fixture(scope="module")
def simple_square_acquisition():
    class SimpleSquareAcquisition(Acquisition):
        def __init__(self):
            pass

        def evaluate(self, x):
            y = -x[:, 0] ** 2 + np.sum(x[:, 1:], axis=1) + 1
            return np.atleast_2d(y).T

        @property
        def has_gradients(self):
            return False

    return SimpleSquareAcquisition()
