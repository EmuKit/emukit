# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Warpings as used by warped Bayesian quadrature models."""

import abc
from typing import Optional

import numpy as np


class Warping(abc.ABC):
    """Base class for a warping as used by a warped Bayesian quadrature model.

    .. seealso::
        * :class:`emukit.quadrature.methods.WarpedBayesianQuadratureModel`
        * :class:`emukit.quadrature.methods.warpings.IdentityWarping`
        * :class:`emukit.quadrature.methods.warpings.SquareRootWarping`

    """

    @abc.abstractmethod
    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: Function values of latent function, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: Function values of integrand, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        raise NotImplementedError

    def update_parameters(self, **new_parameters) -> None:
        """Update the warping parameters.

        :param new_parameters: Contains the parameter names as keys with the new values.
                               An empty dictionary will do nothing.
        """
        for parameter, new_value in new_parameters.items():
            setattr(self, parameter, new_value)


class IdentityWarping(Warping):
    """The identity warping

    .. math::
        w(y) = y.

    """

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return Y

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return Y


class SquareRootWarping(Warping):
    r"""The square root warping.

    .. math::
        w(y)=\begin{cases}
                  c + \frac{1}{2}y^2 & \text{is_inverted is False (default)}\\
                  c - \frac{1}{2}y^2 &\text{otherwise}
              \end{cases},

    where :math:`c` is a constant.

    :param offset: The offset :math:`c` of the warping.
    :param is_inverted: Inverts the warping if ``True``. Default is ``False``.

    """

    def __init__(self, offset: float, is_inverted: Optional[bool] = False):
        self.offset = offset
        self.is_inverted = is_inverted

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.is_inverted:
            return self.offset - 0.5 * (Y * Y)
        else:
            return self.offset + 0.5 * (Y * Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        if self.is_inverted:
            return np.sqrt(2.0 * (self.offset - Y))
        else:
            return np.sqrt(2.0 * (Y - self.offset))
