# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, List
import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def circular_gaussian(mean: np.float = 0., variance: np.float = 1.) -> Tuple[UserFunctionWrapper,
                                                                             List[Tuple[float, float]]]:
    """
    2D toy integrand that is a Gaussian on a circle
    """
    func = lambda x: _circular_gaussian(x, mean, variance)
    integral_bounds = 2 * [(-3., 3.)]
    return UserFunctionWrapper(func), integral_bounds


def _circular_gaussian(x: np.ndarray, mean: np.float, variance: np.float) -> np.ndarray:
    """
    :param x: (num_points, 2)
    :param mean: mean of Gaussian in radius (must be > 0)
    :return: the function values at x, shape (num_points, 1)
    """
    norm_x = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    return norm_x**2 * np.exp(- (norm_x - mean)**2 / (2. * variance)) / np.sqrt(2. * np.pi * variance)
