# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop.user_function import MultiSourceFunctionWrapper


def multi_fidelity_forrester_function(high_fidelity_noise_std_deviation=0, low_fidelity_noise_std_deviation=0):
    """
    Two-level multi-fidelity forrester function where the high fidelity is given by:

    .. math::
        f(x) = (6x - 2)^2 \sin(12x - 4)

    and the low fidelity approximation given by:

    .. math::
        f_{low}(x) = 0.5 f_{high}(x) + 10 (x - 0.5) + 5

    :param high_fidelity_noise_std_deviation: Standard deviation of observation noise on high fidelity observations.
                                              Defaults to zero.
    :param low_fidelity_noise_std_deviation: Standard deviation of observation noise on low fidelity observations.
                                             Defaults to zero.
    :return: Tuple of user function object and parameter space object
    """
    parameter_space = ParameterSpace([ContinuousParameter('x', 0, 1), InformationSourceParameter(2)])
    user_function = MultiSourceFunctionWrapper([
        lambda x: forrester_low(x, low_fidelity_noise_std_deviation),
        lambda x: forrester(x, high_fidelity_noise_std_deviation)])
    return user_function, parameter_space


def forrester_function(noise_standard_deviation=0):
    """
    Forrester function

    .. math::
        f(x) = (6x - 2)^2 \sin(12x - 4)

    :param noise_standard_deviation: Standard deviation of normally distributed observation noise
    :return: Tuple of function and parameter space object
    """
    def forrester_fcn(x):
        return forrester(x, sd=noise_standard_deviation)
    return forrester_fcn, ParameterSpace([ContinuousParameter('x', 0, 1)])


def forrester(x, sd=0):
    """
    Forrester function

    :param x: input vector to be evaluated
    :param sd: standard deviation of noise parameter
    :return: outputs of the function
    """
    x = x.reshape((len(x), 1))
    n = x.shape[0]
    fval = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    if sd == 0:
        noise = np.zeros(n).reshape(n, 1)
    else:
        noise = np.random.normal(0, sd, n).reshape(n, 1)
    return fval.reshape(n, 1) + noise


def forrester_low(x, sd=0):
    """
    Low fidelity forrester function approximation:

    :param x: input vector to be evaluated
    :param sd: standard deviation of observation noise at low fidelity
    :return: outputs of the function
    """
    high_fidelity = forrester(x, 0)
    return 0.5 * high_fidelity + 10 * (x[:, [0]] - 0.5) + 5 + np.random.randn(x.shape[0], 1) * sd
