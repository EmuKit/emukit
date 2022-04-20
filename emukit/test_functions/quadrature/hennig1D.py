# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def hennig1D() -> Tuple[UserFunctionWrapper, List[Tuple[float, float]]]:
    r"""1D toy integrand coined by Philipp Hennig.

    .. math::
        f(x) = e^{-x^2 -\sin^2(3x)}

    :return: The wrapped test function, and the integrals bounds
             (the latter default to [-3, 3]).
    """
    integral_bounds = [(-3.0, 3.0)]
    return UserFunctionWrapper(_hennig1D), integral_bounds


def _hennig1D(x: np.ndarray) -> np.ndarray:
    """
    :param x: Locations for evaluation (num_points, 1).
    :return: The function values at x, shape (num_points, 1).
    """
    return np.exp(-(x**2) - np.sin(3.0 * x) ** 2)
