# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""The product Matern52 kernel embeddings."""

from typing import Union

import numpy as np

from ...quadrature.interfaces.standard_kernels import IProductMatern52
from ..measures import IntegrationMeasure, LebesgueMeasure
from .quadrature_kernels import LebesgueEmbedding, QuadratureProductKernel


class QuadratureProductMatern52(QuadratureProductKernel):
    r"""Base class for a product Matern52 kernel augmented with integrability.

    The kernel is of the form :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{5} r_i + \frac{5}{3} r_i^2) \exp(- \sqrt{5} r_i).

    Above, :math:`d` is the input dimensionality, :math:`r_i =\frac{|x_i - x'_i|}{\lambda_i}`,
    is the scaled distance, :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i`
    is the :math:`i` th element of the ``lengthscales`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IProductMatern52`.
        Each subclass of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern52`
       * :class:`emukit.quadrature.kernels.QuadratureProductKernel`

    :param matern_kernel: The standard EmuKit product Matern52 kernel.
    :param measure: The integration measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        matern_kernel: IProductMatern52,
        measure: IntegrationMeasure,
        variable_names: str,
    ) -> None:
        super().__init__(kern=matern_kernel, measure=measure, variable_names=variable_names)

    @property
    def nu(self) -> float:
        """The smoothness parameter of the kernel."""
        return self.kern.nu

    @property
    def lengthscales(self) -> np.ndarray:
        r"""The lengthscales :math:`\lambda` of the kernel."""
        return self.kern.lengthscales

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        return self.kern.variance


class QuadratureProductMatern52LebesgueMeasure(QuadratureProductMatern52, LebesgueEmbedding):
    """A product Matern52 kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern52`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern52`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param matern_kernel: The standard EmuKit product Matern52 kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern52, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(matern_kernel=matern_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        return {
            "domain": self.measure.domain.bounds[dim],
            "lengthscale": self.lengthscales[dim],
            "normalize": self.measure.is_normalized,
        }

    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s5 = np.sqrt(5)
        first_term = 16 * lengthscale / (3 * s5)
        second_term = (
            -np.exp(s5 * (x - b) / lengthscale)
            / (15 * lengthscale)
            * (8 * s5 * lengthscale**2 + 25 * lengthscale * (b - x) + 5 * s5 * (b - x) ** 2)
        )
        third_term = (
            -np.exp(s5 * (a - x) / lengthscale)
            / (15 * lengthscale)
            * (8 * s5 * lengthscale**2 + 25 * lengthscale * (x - a) + 5 * s5 * (a - x) ** 2)
        )
        return (first_term + second_term + third_term) * normalization

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        c = np.sqrt(5) * (b - a)
        bracket_term = 5 * a**2 - 10 * a * b + 5 * b**2 + 7 * c * lengthscale + 15 * lengthscale**2
        qKq = (2 * lengthscale * (8 * c - 15 * lengthscale) + 2 * np.exp(-c / lengthscale) * bracket_term) / 15
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s5 = np.sqrt(5)
        first_exp = -np.exp(s5 * (x - b) / lengthscale) / (15 * lengthscale)
        first_term = first_exp * (15 * lengthscale - 15 * s5 * (x - b) + 25 / lengthscale * (x - b) ** 2)
        second_exp = -np.exp(s5 * (a - x) / lengthscale) / (15 * lengthscale)
        second_term = second_exp * (-15 * lengthscale + 15 * s5 * (a - x) - 25 / lengthscale * (a - x) ** 2)
        return (first_term + second_term) * normalization
