# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter


def branin_function():
    """
    Two-dimensional Branin, often used as an optimization benchmark.

    Based on: https://www.sfu.ca/~ssurjano/branin.html

    .. math::
        f(\mathbf{x}) = (x_2 - b x_1 ^ 2 + c x_1 - r) ^ 2 + s(1 - t) \cos(x_1) + s

    where:

    .. math::
        b = 5.1 / (4 \pi ^ 2)

        c = 5 /\pi

        r = 6

        s = 10

        t = 1 / (8\pi)
    """

    parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 10), ContinuousParameter('x2', 0, 15)])
    return _branin, parameter_space


def _branin(x):
    """
    :param x: n_points x 2 array of input locations to evaluate
    :return: n_points x 1 array of function evaluations
    """
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    x1 = x[:, 0]
    x2 = x[:, 1]
    y = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return y[:, None]
