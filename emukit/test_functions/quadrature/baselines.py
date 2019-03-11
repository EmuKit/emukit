# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, List
import numpy as np
from scipy.integrate import quad, dblquad


def univariate_approximate_ground_truth_integral(func, integral_bounds: Tuple[float, float]):
    """
    Uses scipy.integrate.quad to estimate the ground truth integral

    :param func: univariate function
    :param integral_bounds: bounds of integral
    :returns: integral estimate, output of scipy.integrate.quad
    """
    lower_bound = integral_bounds[0]
    upper_bound = integral_bounds[1]
    return quad(func, lower_bound, upper_bound)


def bivariate_approximate_ground_truth_integral(func, integral_bounds: List[Tuple[float, float]]):
    """
    Uses scipy.integrate.dblquad to estimate the ground truth integral

    :param func: bivariate function
    :param integral_bounds: bounds of integral
    :returns: integral estimate, output of scipy.integrate.dblquad
    """
    def func_dblquad(x, y):
        z = np.array([x, y])
        z = np.reshape(z, [1, 2])
        return func(z)

    lower_bound = integral_bounds[0][0]
    upper_bound = integral_bounds[0][1]

    return dblquad(func=func_dblquad, a=lower_bound, b=upper_bound, gfun=lambda x: lower_bound,
                   hfun=lambda x: upper_bound)
