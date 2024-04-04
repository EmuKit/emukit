# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def sombrero2D(freq: float = 1.0) -> Tuple[UserFunctionWrapper, List[Tuple[float, float]]]:
    r"""2D Sombrero function.

    .. math::
        f(x) = \frac{\operatorname{sin}(\pi r \omega)}{\pi r \omega}

    where :math:`\omega` is the :attr:`freq` parameter and :math:`r=\|x\|` is the length
    of the input vector :math:`x`.

    :param freq: The frequency of the sombrero (must be > 0, defaults to 1).
    :return: The wrapped test function, and the integrals bounds (the latter defaults to [-3, 3]^2).
    """
    func = lambda x: _sombrero2D(x, freq)
    integral_bounds = 2 * [(-3.0, 3.0)]
    return UserFunctionWrapper(func), integral_bounds


def _sombrero2D(x: np.ndarray, freq: float) -> np.ndarray:
    """
    :param x: Locations for evaluation (num_points, 2).
    :param freq: The frequency of the sombrero (must be > 0).
    :return: The function values at x, shape (num_points, 1).
    """
    r = np.sqrt((x * x).sum(axis=1))
    r_scaled = (np.pi * freq) * r
    result = np.sin(r_scaled) / r_scaled
    result[np.isnan(result)] = 1.0
    return result[:, np.newaxis]
