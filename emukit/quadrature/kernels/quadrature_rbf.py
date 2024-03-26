# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""The RBF kernel embeddings."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union

import numpy as np
from scipy.special import erf

from ...quadrature.interfaces.standard_kernels import IRBF
from ..kernels import GaussianEmbedding, LebesgueEmbedding, QuadratureKernel
from ..measures import GaussianMeasure, IntegrationMeasure, LebesgueMeasure


class QuadratureRBF(QuadratureKernel):
    r"""Base class for an RBF kernel augmented with integrability.

    .. math::
        k(x, x') = \sigma^2 e^{-\frac{1}{2}\sum_{i=1}^{d}r_i^2},

    where :math:`d` is the input dimensionality,
    :math:`r_i = \frac{x_i-x_i'}{\lambda_i}` is the scaled vector difference of dimension :math:`i`,
    :math:`\lambda_i` is the :math:`i` th element of the ``lengthscales`` property
    and :math:`\sigma^2` is the ``variance`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IRBF`.
        Each child of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureKernel`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param measure: The integration measure. ``None`` implies the standard Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        rbf_kernel: IRBF,
        measure: IntegrationMeasure,
        variable_names: str,
    ) -> None:
        super().__init__(kern=rbf_kernel, measure=measure, variable_names=variable_names)

    @property
    def lengthscales(self) -> np.ndarray:
        r"""The lengthscales :math:`\lambda` of the kernel."""
        return self.kern.lengthscales

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        return self.kern.variance

    # rbf-kernel specific helper
    def _scaled_vector_diff(
        self, v1: np.ndarray, v2: np.ndarray, scales: Union[float, np.ndarray] = None
    ) -> np.ndarray:
        r"""Scaled element-wise vector difference between vectors v1 and v2.

        .. math::
            \frac{v_1 - v_2}{\ell \sqrt{2}}

        where :math:`\ell` is the ``scales`` parameter.

        :param v1: First vector.
        :param v2: Second vector, must have same second dimensions as v1.
        :param scales: The scales, default is the lengthscales of the kernel.
        :return: Scaled difference between v1 and v2, same shape as v1 and v2.
        """
        if scales is None:
            scales = self.lengthscales
        return (v1 - v2) / (scales * np.sqrt(2))


class QuadratureRBFLebesgueMeasure(QuadratureRBF, LebesgueEmbedding):
    """An RBF kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBF`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, rbf_kernel: IRBF, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(rbf_kernel=rbf_kernel, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        erf_lo = erf(self._scaled_vector_diff(lb, x2))
        erf_up = erf(self._scaled_vector_diff(ub, x2))
        kernel_mean = (np.sqrt(np.pi / 2.0) * self.lengthscales * (erf_up - erf_lo)).prod(axis=1)
        return (self.variance * self.measure.density) * kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        diff_bounds_scaled = self._scaled_vector_diff(ub, lb)
        exp_term = (np.exp(-(diff_bounds_scaled**2)) - 1.0) / np.sqrt(np.pi)
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled
        qKq = ((2 * np.sqrt(np.pi) * self.lengthscales**2) * (exp_term + erf_term)).prod()
        return (self.variance * self.measure.density**2) * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        exp_lo = np.exp(-self._scaled_vector_diff(x2, lb) ** 2)
        exp_up = np.exp(-self._scaled_vector_diff(x2, ub) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lb, x2))
        erf_up = erf(self._scaled_vector_diff(ub, x2))
        fraction = ((exp_lo - exp_up) / (self.lengthscales * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo))).T
        return self.qK(x2) * fraction


class QuadratureRBFGaussianMeasure(QuadratureRBF, GaussianEmbedding):
    """An RBF kernel augmented with integrability w.r.t. a Gaussian measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBF`
       * :class:`emukit.quadrature.measures.GaussianMeasure`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param measure: A Gaussian measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, rbf_kernel: IRBF, measure: GaussianMeasure, variable_names: str = "") -> None:
        super().__init__(rbf_kernel=rbf_kernel, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        lengthscales = scale_factor * self.lengthscales
        sigma2 = self.measure.variance
        mu = self.measure.mean
        factor = np.sqrt(lengthscales**2 / (lengthscales**2 + sigma2)).prod()
        scaled_norm_sq = np.power(self._scaled_vector_diff(x2, mu, np.sqrt(lengthscales**2 + sigma2)), 2).sum(axis=1)
        return (self.variance * factor) * np.exp(-scaled_norm_sq).reshape(1, -1)

    def qKq(self) -> float:
        lengthscales = self.lengthscales
        qKq = np.sqrt(lengthscales**2 / (lengthscales**2 + 2 * self.measure.variance)).prod()
        return self.variance * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        scaled_diff = (x2 - self.measure.mean) / (self.lengthscales**2 + self.measure.variance)
        return -self.qK(x2) * scaled_diff.T
