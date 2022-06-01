"""The RBF kernel embeddings."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Union

import numpy as np
from scipy.special import erf

from ...quadrature.interfaces.standard_kernels import IRBF
from ..kernels import QuadratureKernel
from ..measures import BoxDomain, IntegrationMeasure, IsotropicGaussianMeasure, UniformMeasure
from ..typing import BoundsType


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
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param measure: The integration measure. ``None`` implies the standard Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        rbf_kernel: IRBF,
        integral_bounds: Optional[BoundsType],
        measure: Optional[IntegrationMeasure],
        variable_names: str = "",
    ) -> None:
        super().__init__(
            kern=rbf_kernel, integral_bounds=integral_bounds, measure=measure, variable_names=variable_names
        )

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


class QuadratureRBFLebesgueMeasure(QuadratureRBF):
    """An RBF kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBF`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, rbf_kernel: IRBF, integral_bounds: BoundsType, variable_names: str = "") -> None:
        super().__init__(
            rbf_kernel=rbf_kernel, integral_bounds=integral_bounds, measure=None, variable_names=variable_names
        )

    def qK(self, x2: np.ndarray) -> np.ndarray:
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))
        kernel_mean = (np.sqrt(np.pi / 2.0) * self.lengthscales * (erf_up - erf_lo)).prod(axis=1)
        return self.variance * kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        diff_bounds_scaled = self._scaled_vector_diff(upper_bounds, lower_bounds)
        exp_term = (np.exp(-(diff_bounds_scaled**2)) - 1.0) / np.sqrt(np.pi)
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled
        qKq = ((2 * np.sqrt(np.pi) * self.lengthscales**2) * (exp_term + erf_term)).prod()
        return self.variance * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        exp_lo = np.exp(-self._scaled_vector_diff(x2, lower_bounds) ** 2)
        exp_up = np.exp(-self._scaled_vector_diff(x2, upper_bounds) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))

        fraction = ((exp_lo - exp_up) / (self.lengthscales * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo))).T

        return self.qK(x2) * fraction


class QuadratureRBFIsoGaussMeasure(QuadratureRBF):
    """An RBF kernel augmented with integrability w.r.t. an isotropic Gaussian measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBF`
       * :class:`emukit.quadrature.measures.IsotropicGaussianMeasure`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param measure: A Gaussian measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, rbf_kernel: IRBF, measure: IsotropicGaussianMeasure, variable_names: str = "") -> None:
        super().__init__(rbf_kernel=rbf_kernel, integral_bounds=None, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        ells = scale_factor * self.lengthscales
        sigma2 = self.measure.variance
        mu = self.measure.mean
        factor = np.sqrt(ells**2 / (ells**2 + sigma2)).prod()
        scaled_norm_sq = np.power(self._scaled_vector_diff(x2, mu, np.sqrt(ells**2 + sigma2)), 2).sum(axis=1)
        return (self.variance * factor) * np.exp(-scaled_norm_sq).reshape(1, -1)

    def qKq(self) -> float:
        ells = self.lengthscales
        qKq = np.sqrt(ells**2 / (ells**2 + 2 * self.measure.variance)).prod()
        return self.variance * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        scaled_diff = (x2 - self.measure.mean) / (self.lengthscales**2 + self.measure.variance)
        return -self.qK(x2) * scaled_diff.T


class QuadratureRBFUniformMeasure(QuadratureRBF):
    """An RBF kernel augmented with integrability w.r.t. a uniform measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBF`
       * :class:`emukit.quadrature.measures.UniformMeasure`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param measure: A D-dimensional uniform measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        rbf_kernel: IRBF,
        integral_bounds: Optional[BoundsType],
        measure: UniformMeasure,
        variable_names: str = "",
    ):
        super().__init__(
            rbf_kernel=rbf_kernel, integral_bounds=integral_bounds, measure=measure, variable_names=variable_names
        )

        # construct bounds that are used in the computation of the kernel integrals. The lower bounds are the max of
        # the lower bounds of integral and measure. The upper bounds are the min of the upper bounds of integral and
        # measure, i.e., the resulting bounds are the overlap over the integral bounds and the measure bounds.
        if integral_bounds is None:
            bounds = measure.get_box()
        else:
            bounds = [(max(ib[0], mb[0]), min(ib[1], mb[1])) for (ib, mb) in zip(integral_bounds, measure.get_box())]

        # checks if lower bounds are smaller than upper bounds.
        for (lb_d, ub_d) in bounds:
            if lb_d >= ub_d:
                raise ValueError(
                    "Upper bound of relevant integration domain must be larger than lower bound. Found a "
                    "pair containing ({}, {}).".format(lb_d, ub_d)
                )
        self._bounds_list_for_kernel_integrals = bounds
        self.reasonable_box = BoxDomain(name=variable_names, bounds=bounds)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))
        kernel_mean = (np.sqrt(np.pi / 2.0) * self.lengthscales * (erf_up - erf_lo)).prod(axis=1)
        return (self.variance * self.measure._density) * kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        diff_bounds_scaled = self._scaled_vector_diff(upper_bounds, lower_bounds)
        exp_term = (np.exp(-(diff_bounds_scaled**2)) - 1.0) / np.sqrt(np.pi)
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled
        qKq = ((2 * np.sqrt(np.pi) * self.lengthscales**2) * (exp_term + erf_term)).prod()
        return (self.variance * self.measure._density**2) * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        exp_lo = np.exp(-self._scaled_vector_diff(x2, lower_bounds) ** 2)
        exp_up = np.exp(-self._scaled_vector_diff(x2, upper_bounds) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))

        fraction = ((exp_lo - exp_up) / (self.lengthscales * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo))).T

        return self.qK(x2) * fraction
