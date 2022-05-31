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
        k(x, x') = \sigma^2 e^{-\frac{1}{2}\frac{\|x-x'\|^2}{\lambda^2}},

    where :math:`\sigma^2` is the ``variance`` property and :math:`\lambda` is the
    ``lengthscale`` property.

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
    def lengthscale(self) -> Union[np.ndarray, float]:
        r"""The lengthscale(s) :math:`\lambda` of the kernel."""
        return self.kern.lengthscale

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        return self.kern.variance

    # rbf-kernel specific helper
    def _scaled_vector_diff(self, v1: np.ndarray, v2: np.ndarray, scale: float = None) -> np.ndarray:
        r"""Scaled element-wise vector difference between vectors v1 and v2.

        .. math::
            \frac{v_1 - v_2}{\lambda \sqrt{2}}

        name mapping:
            \lambda: self.kern.lengthscale

        :param v1: First vector.
        :param v2: Second vector, must have same second dimensions as v1.
        :param scale: The scale, default is the lengthscale of the kernel
        :return: Scaled difference between v1 and v2, same shape as v1 and v2.
        """
        if scale is None:
            scale = self.lengthscale
        return (v1 - v2) / (scale * np.sqrt(2))


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
        kernel_mean = self.variance * (self.lengthscale * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo)).prod(axis=1)

        return kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        prefac = self.variance * (2.0 * self.lengthscale**2) ** self.input_dim
        diff_bounds_scaled = self._scaled_vector_diff(upper_bounds, lower_bounds)
        exp_term = np.exp(-(diff_bounds_scaled**2)) - 1.0
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled * np.sqrt(np.pi)

        return float(prefac * (exp_term + erf_term).prod())

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        exp_lo = np.exp(-self._scaled_vector_diff(x2, lower_bounds) ** 2)
        exp_up = np.exp(-self._scaled_vector_diff(x2, upper_bounds) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))

        fraction = ((exp_lo - exp_up) / (self.lengthscale * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo))).T

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
        lengthscale = scale_factor * self.lengthscale
        det_factor = (self.measure.variance / lengthscale**2 + 1) ** (self.input_dim / 2)
        scale = np.sqrt(lengthscale**2 + self.measure.variance)
        scaled_vector_diff = self._scaled_vector_diff(x2, self.measure.mean, scale)
        kernel_mean = (self.variance / det_factor) * np.exp(-np.sum(scaled_vector_diff**2, axis=1))
        return kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        factor = (2 * self.measure.variance / self.lengthscale**2 + 1) ** (self.input_dim / 2)
        result = self.variance / factor
        return float(result)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        qK_x = self.qK(x2)
        factor = 1.0 / (self.lengthscale**2 + self.measure.variance)
        return -(qK_x * factor) * (x2 - self.measure.mean).T


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
        kernel_mean = self.variance * (self.lengthscale * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo)).prod(axis=1)

        return kernel_mean.reshape(1, -1) * self.measure._density

    def qKq(self) -> float:
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        prefac = self.variance * (2.0 * self.lengthscale**2) ** self.input_dim
        diff_bounds_scaled = self._scaled_vector_diff(upper_bounds, lower_bounds)
        exp_term = np.exp(-(diff_bounds_scaled**2)) - 1.0
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled * np.sqrt(np.pi)

        return float(prefac * (exp_term + erf_term).prod()) * self.measure._density**2

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        exp_lo = np.exp(-self._scaled_vector_diff(x2, lower_bounds) ** 2)
        exp_up = np.exp(-self._scaled_vector_diff(x2, upper_bounds) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))

        fraction = ((exp_lo - exp_up) / (self.lengthscale * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo))).T

        return self.qK(x2) * fraction
