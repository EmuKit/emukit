# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, List
import numpy as np

from emukit.core.loop.user_function import UserFunctionWrapper


def sombrero2D(freq: float = 1.) -> Tuple[UserFunctionWrapper, List[Tuple[float, float]]]:
    """
    2D Sombrero function

    Evaluates the function at x.
    .. math::

        \frac{\sin(\pi r freq)}{\pi r freq}

    """
    func = lambda x: _sombrero2D(x, freq)
    integral_bounds = 2 * [(-3., 3.)]
    return UserFunctionWrapper(func), integral_bounds


def _sombrero2D(x: np.ndarray, freq: float) -> np.ndarray:
    """
    :param x: (num_points, 2)
    :param freq: frequency of the sombrero
    :return: the function values at x, shape (num_points, 1)
    """
    r = np.sqrt((x*x).sum(axis=1))
    r_scaled = (np.pi * freq) * r
    result = np.sin(r_scaled) / r_scaled
    result[np.isnan(result)] = 1.
    return result[:, np.newaxis]
