# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Callable
import numpy as np
from scipy.integrate import quad

from emukit.core.loop.user_function import UserFunctionWrapper


def hennig1D() -> Tuple[UserFunctionWrapper, Callable]:
    """
    1D toy integrand coined by Philipp Hennig

    Evaluates the function at x.
    .. math::

        e^{-x^2 -\sin^2(3x)}

    """
    return UserFunctionWrapper(_hennig1D), _hennig1D


def _hennig1D(x: np.ndarray) -> np.ndarray:
    """
    :param x: (num_points, 1)
    :return: the function values at x, shape (num_points, 1)
    """
    return np.exp(- x**2 - np.sin(3. * x)**2)


def circular_gaussian(mean: np.float = 0., variance: np.float = 1.) -> Tuple[UserFunctionWrapper, Callable]:
    """ 2D toy integrand that is a Gaussian on a circle
    """
    return UserFunctionWrapper(_circular_gaussian), lambda x: _circular_gaussian(x, mean, variance)


def _circular_gaussian(x: np.ndarray, mean: np.float, variance: np.float):
    """
    :param x: (num_points, 2)
    :param mean: mean of Gaussian in radius (must be > 0)
    """
    norm_x = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    return norm_x**2 * np.exp(- (norm_x - mean)**2 / (2. * variance)) / np.sqrt(2. * np.pi * variance)


# these integrators help with testing and illustrations
def univariate_approximate_ground_truth_integral(func, integral_bounds: Tuple[np.float, np.float]):
    """
    Uses scipy.integrate.quad to estimate the ground truth

    :param func: univariate function
    :param integral_bounds: bounds of integral
    :returns: integral estimate, output of scipy.integrate.quad
    """
    lower_bound = integral_bounds[0]
    upper_bound = integral_bounds[1]
    return quad(func, lower_bound, upper_bound)
