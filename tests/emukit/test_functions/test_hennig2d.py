# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions.quadrature import hennig2D


def test_hennig2D_return_shape():
    func, _ = hennig2D()
    x = np.ones((3, 2))
    result = func.f(x)
    assert result.ndim == 2
    assert result.shape == (3, 1)
