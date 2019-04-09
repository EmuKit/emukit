# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from emukit.core import ParameterSpace, ContinuousParameter


def sixhumpcamel_function():
    """
    Two-dimensional SixHumpCamel function, often used as an optimization benchmark.

    Based on: https://www.sfu.ca/~ssurjano/camel6.html

    .. math::
        f(\mathbf{x}) = \left(4-2.1x_1^2 = \frac{x_1^4}{3} \right)x_1^2 + x_1x_2 + (-4 +4x_2^2)x_2^2

    """

    parameter_space = ParameterSpace([ContinuousParameter('x1', -2, 2), ContinuousParameter('x2', -1, 1)])
    return _sixhumpcamel, parameter_space


def _sixhumpcamel(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    y = term1 + term2 + term3
    return y[:,None]
