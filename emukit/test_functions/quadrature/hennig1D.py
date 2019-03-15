# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, List
import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def hennig1D() -> Tuple[UserFunctionWrapper, List[Tuple[float, float]]]:
    """
    1D toy integrand coined by Philipp Hennig

    .. math::

        e^{-x^2 -\sin^2(3x)}

    :return: the wrapped test function, and the integrals bounds (defaults to interval [-3, 3]).
    """
    integral_bounds = [(-3., 3.)]
    return UserFunctionWrapper(_hennig1D), integral_bounds


def _hennig1D(x: np.ndarray) -> np.ndarray:
    """
    :param x: locations for evaluation (num_points, 1)
    :return: the function values at x, shape (num_points, 1)
    """
    return np.exp(- x**2 - np.sin(3. * x)**2)
