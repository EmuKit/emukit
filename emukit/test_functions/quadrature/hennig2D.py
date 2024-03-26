# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def hennig2D() -> Tuple[UserFunctionWrapper, List[Tuple[float, float]]]:
    r"""2D toy integrand coined by Philipp Hennig.

    .. math::
        f(x) = e^{-x'Sx -\sin(3\|x\|^2)}

    :return: The wrapped test function, and the integrals bounds
             (the latter default to [-3, 3]^2).
    """
    integral_bounds = 2 * [(-3.0, 3.0)]
    return UserFunctionWrapper(_hennig2D), integral_bounds


def _hennig2D(x: np.ndarray, S: np.ndarray = None) -> np.ndarray:
    """
    :param x: Locations for evaluation (num_points, 2).
    :return: The function values at x, shape (num_points, 2).
    """
    if S is None:
        S = np.array([[1, 0.5], [0.5, 1]])
    f = np.exp(-np.sin(3 * np.sum(x**2, axis=1)) - np.sum((x @ S) * x, axis=1))
    return np.reshape(f, [x.shape[0], 1])
