from typing import Tuple

import numpy as np

from ...core.loop.user_function import MultiSourceFunctionWrapper
from ...core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_park_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""

    High fidelity is given by:

    .. math::
        f_{high}(\mathbf{x}) = \frac{x_1}{2}\left[ \sqrt{1 + \left( x_2 + x_3^2 \right) \frac{x_4}{x_1^2}} - 1\right] +
        \left( x_1 + 3 x_4 \right) \exp \left[ 1 + \sin x_3 \right]

    Low fidelity is given by:

    .. math::
        f_{low}(\mathbf{x}) = \left[ 1 + \frac{\sin x_1}{10} \right] f_{high}(\mathbf{x}) -
        2x_1 + x_2 ^ 2 + x_3 ^2 + 0.5

    The input domain is given by:

    .. math::
        \mathbf{x}_i \in (0, 1)

    Reference: https://www.sfu.ca/~ssurjano/park91a.html
    """

    def park_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        tmp = 1 + (x2 + x3) * (x4 / x1**2)
        return ((x1 / 2) * (np.sqrt(tmp) - 1) + (x1 + 3 * x4) * np.exp(1 + np.sin(x3)))[:, None]

    def park_low(x):
        f_high = park_high(x)

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        return ((1 + np.sin(x1) / 10) * f_high.flatten() - 2 * x1 + x2**2 + x3**2 + 0.5)[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            ContinuousParameter('x3', 0., 1.), ContinuousParameter('x4', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([park_low, park_high]), space
