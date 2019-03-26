from typing import Tuple

import numpy as np

from ...core.loop.user_function import MultiSourceFunctionWrapper
from ...core import ParameterSpace, ContinuousParameter, InformationSourceParameter


def multi_fidelity_hartmann_3d() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    The function is given by:

    .. math::
        f(x, \alpha) = -\sum_{i=1}^{4} \alpha_i \exp \left( -\sum_{j=1}^{3} A_{i,j}\left( x_j - P_{i, j} \right)^2 \right)

    where

    .. math::
        \mathbf{A} = \begin{bmatrix}
        3.0 & 10 & 30 \\
        0.1 & 10 & 35 \\
        3.0 & 10 & 30 \\
        0.1 & 10 & 35
        \end{bmatrix}

    .. math::
        \mathbf{P} = 10^{-4} \begin{bmatrix}
        3689 & 1170 & 2673 \\
        4699 & 4387 & 7470 \\
        1091 & 8732 & 5547 \\
        381 & 5743 & 8828
        \end{bmatrix}

    The high fidelity function is given by setting:

    .. math::
        \alpha = (1.0, 1.2, 3.0, 3.2)^T

    The middle fidelity is given by setting:

    .. math::
        \alpha = (1.01, 1.19, 2.9, 3.3)^T

    The low fidelity is given by setting:

    .. math::
        \alpha = (1.02, 1.18, 2.8, 3.4)^T

    The domain is given by:

    .. math::
        \mathbf{x}_i \in (0, 1)

    Reference: https://www.sfu.ca/~ssurjano/hart3.html

    :return: Tuple of MultiSourceFunctionWrapper and ParameterSpace
    """
    A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    delta = np.array([0.01, -0.01, -0.1, 0.1])

    def high(x):

        res = 0
        for i in range(4):
            temp = 0
            for j in range(3):
                temp -= A[i][j] * np.power(x[:, j] - P[i][j], 2)
            res += alpha[i] * np.exp(temp)

        return res[:, None]

    def medium(x):

        alpha_m = alpha + delta

        res = 0
        for i in range(4):
            temp = 0
            for j in range(3):
                temp -= A[i][j] * np.power(x[:, j] - P[i][j], 2)
            res += alpha_m[i] * np.exp(temp)

        return res[:, None]

    def low(x):

        alpha_l = alpha + 2 * delta

        res = 0
        for i in range(4):
            temp = 0
            for j in range(3):
                temp -= A[i][j] * np.power(x[:, j] - P[i][j], 2)
            res += alpha_l[i] * np.exp(temp)

        return res[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            ContinuousParameter('x3', 0., 1.), InformationSourceParameter(3)])

    fcn_wrapper = MultiSourceFunctionWrapper([low, medium, high])

    return fcn_wrapper, space
