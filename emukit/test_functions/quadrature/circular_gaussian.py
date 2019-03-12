# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, List
import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def circular_gaussian(mean: np.float = 0., variance: np.float = 1.) -> Tuple[UserFunctionWrapper,
                                                                             List[Tuple[float, float]]]:
    """
    2D toy integrand that is a Gaussian on a circle

    .. math::

        (2\pi \sigma^2)^{-0.5} r^2 e^{-\frac{(r - \mu)^2}{2 \sigma^2}}

    :param mean: the mean of the circular Gaussian in units of radius (must be >= 0, defaults to 0)
    :param variance: variance of the Gaussian (must be > 0, defaults to 1.)
    :return: the wrapped test function, and the integrals bounds (defaults to area [-3, 3]^2).
    """
    func = lambda x: _circular_gaussian(x, mean, variance)
    integral_bounds = 2 * [(-3., 3.)]
    return UserFunctionWrapper(func), integral_bounds


def _circular_gaussian(x: np.ndarray, mean: np.float, variance: np.float) -> np.ndarray:
    """
    :param x: locations for evaluation (num_points, 2)
    :param mean: mean of Gaussian in units of radius (must be >= 0)
    :param variance: variance of the Gaussian (must be > 0)
    :return: the function values at x, shape (num_points, 1)
    """
    norm_x = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    return norm_x**2 * np.exp(- (norm_x - mean)**2 / (2. * variance)) / np.sqrt(2. * np.pi * variance)
