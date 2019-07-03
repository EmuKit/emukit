from typing import Tuple

import numpy as np

from ..branin import _branin
from ...core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from ...core.loop.user_function import MultiSourceFunctionWrapper


def multi_fidelity_branin_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Two-dimensional Branin, often used as an optimization benchmark but here modified to be used as a multi-fidelity
    benchmark

    Based on: https://www.sfu.ca/~ssurjano/branin.html

    High fidelity given by:

    .. math::
        f(\mathbf{x}) = (x_2 - b x_1 ^ 2 + c x_1 - r) ^ 2 + s(1 - t) \cos(x_1) + s

    Middle fidelity given by:

    .. math::
        f_{m}(\mathbf{x}) = \sqrt{f_{high}(\mathbf{x} - 2)} + \frac{2(x_1 - 0.5) - 3(3x_2 - 1) - 1}{100}

    Low fidelity given by:

    .. math::
        f_{m}(1.2(\mathbf{x} + 2)) - \frac{3 x_2 + 1}{100}


    where:

    .. math::
        b = 5.1 / (4 \pi ^ 2)

        c = 5 /\pi

        r = 6

        s = 10

        t = 1 / (8\pi)
    """

    def branin_medium_fidelity(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        result = (10.0 * np.sqrt(_branin(x - 2.0)[:, 0]) + 2.0 * (x1 - 0.5) - 3.0 * (
                    3.0 * x2 - 1.0) - 1.0) / 100.
        return result[:, None]

    def branin_low_fidelity(x):
        x2 = x[:, 1]
        result = (branin_medium_fidelity(1.2 * (x + 2.0))[:, 0] * 100. - 3.0 * x2 + 1.0) / 100.
        return result[:, None]

    parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 10), ContinuousParameter('x2', 0, 15),
                                      InformationSourceParameter(3)])

    branin_high = lambda x: _branin(x)/100
    return MultiSourceFunctionWrapper([branin_low_fidelity, branin_medium_fidelity, branin_high]), parameter_space
