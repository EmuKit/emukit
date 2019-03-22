# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

import numpy as np

from ...core import ContinuousParameter, InformationSourceParameter, ParameterSpace
from ...core.loop.user_function import MultiSourceFunctionWrapper


def multi_fidelity_borehole_function(high_noise_std_deviation: float = 0, low_noise_std_deviation: float = 0) \
        -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    """
    Two level borehole function.

    The Borehole function models water flow through a borehole. Its simplicity and quick evaluation makes it a commonly
    used function for testing a wide variety of methods in computer experiments.

    See reference for equations:
    https://www.sfu.ca/~ssurjano/borehole.html

    :param high_noise_std_deviation: Standard deviation of Gaussian observation noise on high fidelity observations.
                                     Defaults to zero.
    :param low_noise_std_deviation: Standard deviation of Gaussian observation noise on low fidelity observations.
                                     Defaults to zero.
    :return: Tuple of user function object and parameter space
    """
    parameter_space = ParameterSpace([
        ContinuousParameter('borehole_radius', 0.05, 0.15),
        ContinuousParameter('radius_of_influence', 100, 50000),
        ContinuousParameter('upper_aquifer_transmissivity', 63070, 115600),
        ContinuousParameter('upper_aquifer_head', 990, 1110),
        ContinuousParameter('lower_aquifer_transmissivity', 63.1, 116),
        ContinuousParameter('lower_aquifer_head', 700, 820),
        ContinuousParameter('borehole_length', 1120, 1680),
        ContinuousParameter('hydraulic_conductivity', 9855, 12045),
        InformationSourceParameter(2)])

    user_function = MultiSourceFunctionWrapper([
        lambda x: _borehole_low(x, low_noise_std_deviation),
        lambda x: _borehole_high(x, high_noise_std_deviation)])

    return user_function, parameter_space


def _borehole_high(x, sd=0):
    """
    High fidelity version of borehole function

    Reference:
    https://www.sfu.ca/~ssurjano/borehole.html
    """

    numerator = 2 * np.pi * x[:, 2] * (x[:, 3] - x[:, 5])
    ln_r_rw = np.log(x[:, 1] / x[:, 0])
    denominator = ln_r_rw * \
        (x[:, 2] / x[:, 4] + 1 + (2 * x[:, 6] * x[:, 2]) /
         (x[:, 0]**2 * x[:, 7] * ln_r_rw))
    return (numerator / denominator)[:, None] + np.random.randn(x.shape[0], 1) * sd


def _borehole_low(x, sd=0):
    """
    Low fidelity version of borehole function

    Reference:
    https://www.sfu.ca/~ssurjano/borehole.html
    """

    numerator = 5 * x[:, 2] * (x[:, 3] - x[:, 5])
    ln_r_rw = np.log(x[:, 1] / x[:, 0])
    denominator = ln_r_rw * \
        (x[:, 2] / x[:, 4] + 1.5 + (2 * x[:, 6] * x[:, 2]) /
         (x[:, 0]**2 * x[:, 7] * ln_r_rw))
    return (numerator / denominator)[:, None] + np.random.randn(x.shape[0], 1) * sd
