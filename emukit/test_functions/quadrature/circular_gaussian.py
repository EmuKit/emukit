# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def circular_gaussian(
    mean: float = 0.0, variance: float = 1.0
) -> Tuple[UserFunctionWrapper, List[Tuple[float, float]]]:
    r"""2D toy integrand that is a Gaussian on a circle.

    .. math::
        f(x) = (2\pi \sigma^2)^{-\frac{1}{2}} r^2 e^{-\frac{(r - \mu)^2}{2 \sigma^2}}

    where :math:`\sigma^2` is the :attr:`variance` attribute,
    :math:`\mu` is the :attr:`mean` attribute
    and :math:`r = \|x\|` is the length of the input :math:`x`.

    :param mean: The mean of the circular Gaussian in units of radius (must be >= 0, defaults to 0).
    :param variance: The variance of the Gaussian (must be > 0, defaults to 1.).
    :return: The wrapped test function, and the integrals bounds (the latter defaults to [-3, 3]^2).
    """
    func = lambda x: _circular_gaussian(x, mean, variance)
    integral_bounds = 2 * [(-3.0, 3.0)]
    return UserFunctionWrapper(func), integral_bounds


def _circular_gaussian(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """
    :param x: Locations for evaluation (num_points, 2).
    :param mean: The mean of Gaussian in units of radius (must be >= 0).
    :param variance: The variance of the Gaussian (must be > 0).
    :return: The function values at x, shape (num_points, 1).
    """
    norm_x = np.sqrt((x**2).sum(axis=1, keepdims=True))
    return norm_x**2 * np.exp(-((norm_x - mean) ** 2) / (2.0 * variance)) / np.sqrt(2.0 * np.pi * variance)
