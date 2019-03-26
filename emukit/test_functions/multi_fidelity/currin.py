from typing import Tuple

import numpy as np

from ...core.loop.user_function import MultiSourceFunctionWrapper
from ...core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_currin_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""

    High fidelity function is given by:

    .. math::
        f_{high}(\mathbf{x}) = \left[ 1 - \exp \left(-\frac{1}{2x_2}\right) \right]
        \frac{2300x_1^3 + 1900x_1^2 + 2092x_1 + 60}{100x_1^3+500x_1^2 + 4x_1 + 20}

    Low fidelity function given by:

    .. math::
        f_{low}(\mathbf{x}) = \frac{1}{4} \left[ f_{high}(x_1 + 0.05, x_2 + 0.05) + f_{high}(x_1 + 0.05, \max (0, x_2 - 0.05)) \\
        +  f_{high}(x_1 - 0.05, x_2 + 0.05) + f_{high}\left(x_1 - 0.05, \max \left(0, x_2 - 0.05\right)\right) \right]

    Input domain:

    .. math::
        \mathbf{x}_i \in [0, 1]

    Reference: https://www.sfu.ca/~ssurjano/curretal88exp.html
    """
    def high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return (1 - np.exp(-0.5 / x2) * ((2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60)
                                     / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)))[:, None]

    def low(x):
        return (0.25 * high(np.stack([x[:, 0] + 0.05, x[:, 1] + 0.05], axis=1)) +
                0.25 * high(np.stack([x[:, 0] + 0.05, np.maximum(0, x[:, 1] - 0.05)], axis=1)) +
                0.25 * high(np.stack([x[:, 0] - 0.05, x[:, 1] + 0.05], axis=1)) +
                0.25 * high(np.stack([x[:, 0] - 0.05, np.maximum(0, x[:, 1] - 0.05)], axis=1)))

    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space
