# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.special import erf
from typing import List, Tuple, Optional

from .quadrature_kernels import QuadratureKernel
from ...quadrature.interfaces.standard_kernels import IRBF
from ...quadrature.kernels.integration_measures import IntegrationMeasure, IsotropicGaussianMeasure, UniformMeasure
from ...quadrature.kernels.bounds import BoxBounds


class QuadratureRBF(QuadratureKernel):
    """
    Augments an RBF kernel with integrability

    Note 1: each standard kernel goes with a corresponding quadrature kernel, in this case the standard rbf kernel.
    Note 2: each child of this class implements a unique measure-integralBounds pair
    """

    def __init__(self, rbf_kernel: IRBF, integral_bounds: Optional[List[Tuple[float, float]]],
                 measure: Optional[IntegrationMeasure], variable_names: str='') -> None:
        """
        :param rbf_kernel: standard emukit rbf-kernel
        :param integral_bounds: defines the domain of the integral. List of D tuples, where D is the dimensionality
        of the integral and the tuples contain the lower and upper bounds of the integral
        i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]. None for infinite bounds
        :param measure: an integration measure. None means the standard Lebesgue measure is used.
        :param variable_names: the (variable) name(s) of the integral
        """

        super().__init__(kern=rbf_kernel, integral_bounds=integral_bounds, measure=measure, variable_names=variable_names)

    @property
    def lengthscale(self):
        return self.kern.lengthscale

    @property
    def variance(self):
        return self.kern.variance

    def qK(self, x2: np.ndarray) -> np.ndarray:
        """
        RBF kernel with the first component integrated out aka kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x2, shape (1, N)
        """
        raise NotImplementedError

    def Kq(self, x1: np.ndarray) -> np.ndarray:
        """
        RBF kernel with the second component integrated out aka kernel mean

        :param x1: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x1, shape (N, 1)
        """
        return self.qK(x1).T

    def qKq(self) -> float:
        """
        RBF kernel integrated over both arguments x1 and x2

        :returns: double integrated kernel
        """
        raise NotImplementedError

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in first argument) evaluated at x2
        :param x2: points at which to evaluate, shape (n_point N, input_dim)
        :return: the gradient with shape (input_dim, N)
        """
        raise NotImplementedError

    def dKq_dx(self, x1: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in second argument) evaluated at x1
        :param x1: points at which to evaluate, shape (n_point N, input_dim)
        :return: the gradient with shape (N, input_dim)
        """
        return self.dqK_dx(x1).T

    # rbf-kernel specific helper
    def _scaled_vector_diff(self, v1: np.ndarray, v2: np.ndarray, scale: float=None) -> np.ndarray:
        """
        Scaled element-wise vector difference between vectors v1 and v2

        .. math::
            \frac{v_1 - v_2}{\lambda \sqrt{2}}

        name mapping:
            \lambda: self.kern.lengthscale

        :param v1: first vector
        :param v2: second vector, must have same second dimensions as v1
        :param scale: the scale, default is the lengthscale of the kernel
        :return: scaled difference between v1 and v2, np.ndarray with unchanged dimensions

        """
        if scale is None:
            scale = self.lengthscale
        return (v1 - v2) / (scale * np.sqrt(2))


class QuadratureRBFLebesgueMeasure(QuadratureRBF):
    """
    And RBF kernel with integrability over the standard Lebesgue measure. Can only be used with finite integral bounds.

    Note that each standard kernel goes with a corresponding quadrature kernel, in this case standard rbf kernel.
    """

    def __init__(self, rbf_kernel: IRBF, integral_bounds: List[Tuple[float, float]], variable_names: str='') -> None:
        """
        :param rbf_kernel: standard emukit rbf-kernel
        :param integral_bounds: defines the domain of the integral. List of D tuples, where D is the dimensionality
        of the integral and the tuples contain the lower and upper bounds of the integral
        i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        :param variable_names: the (variable) name(s) of the integral
        """
        super().__init__(rbf_kernel=rbf_kernel, integral_bounds=integral_bounds, measure=None,
                         variable_names=variable_names)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        """
        RBF kernel with the first component integrated out aka kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x2, shape (1, N)
        """
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))
        kernel_mean = self.variance * (self.lengthscale * np.sqrt(np.pi / 2.) * (erf_up - erf_lo)).prod(axis=1)

        return kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        """
        RBF kernel integrated over both arguments x1 and x2

        :returns: double integrated kernel
        """
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        prefac = self.variance * (2. * self.lengthscale**2)**self.input_dim
        diff_bounds_scaled = self._scaled_vector_diff(upper_bounds, lower_bounds)
        exp_term = np.exp(-diff_bounds_scaled**2) - 1.
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled * np.sqrt(np.pi)

        return float(prefac * (exp_term + erf_term).prod())

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in first argument) evaluated at x2
        :param x2: points at which to evaluate, shape (n_point N, input_dim)
        :return: the gradient with shape (input_dim, N)
        """
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        exp_lo = np.exp(- self._scaled_vector_diff(x2, lower_bounds) ** 2)
        exp_up = np.exp(- self._scaled_vector_diff(x2, upper_bounds) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))

        fraction = ((exp_lo - exp_up) / (self.lengthscale * np.sqrt(np.pi / 2.) * (erf_up - erf_lo))).T

        return self.qK(x2) * fraction


class QuadratureRBFIsoGaussMeasure(QuadratureRBF):
    """
    Augments an RBF kernel with integrability

    Note that each standard kernel goes with a corresponding quadrature kernel, in this case standard rbf kernel.
    """

    def __init__(self, rbf_kernel: IRBF, measure: IsotropicGaussianMeasure, variable_names: str='') -> None:
        """
        :param rbf_kernel: standard emukit rbf-kernel
        :param measure: a Gaussian measure
        :param variable_names: the (variable) name(s) of the integral
        """
        super().__init__(rbf_kernel=rbf_kernel, integral_bounds=None, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray, scale_factor: float = 1.) -> np.ndarray:
        """
        RBF kernel with the first component integrated out aka kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :param scale_factor: scales the lengthscale of the RBF kernel with the multiplicative factor.
        :returns: kernel mean at location x2, shape (1, N)
        """
        lengthscale = scale_factor * self.lengthscale
        det_factor = (self.measure.variance / lengthscale ** 2 + 1) ** (self.input_dim / 2)
        scale = np.sqrt(lengthscale ** 2 + self.measure.variance)
        scaled_vector_diff = self._scaled_vector_diff(x2, self.measure.mean, scale)
        kernel_mean = (self.variance / det_factor) * np.exp(- np.sum(scaled_vector_diff ** 2, axis=1))
        return kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        """
        RBF kernel integrated over both arguments x1 and x2

        :returns: double integrated kernel
        """
        factor = (2 * self.measure.variance / self.lengthscale ** 2 + 1) ** (self.input_dim / 2)
        result = self.variance / factor
        return float(result)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in first argument) evaluated at x2

        :param x2: points at which to evaluate, shape (n_point N, input_dim)
        :return: the gradient with shape (input_dim, N)
        """
        qK_x = self.qK(x2)
        factor = 1. / (self.lengthscale ** 2 + self.measure.variance)
        return - (qK_x * factor) * (x2 - self.measure.mean).T


class QuadratureRBFUniformMeasure(QuadratureRBF):
    """
    And RBF kernel with integrability over a uniform measure. Can be used with finite as well as infinite integral
    bounds.

    Note that each standard kernel goes with a corresponding quadrature kernel, in this case standard rbf kernel.
    """

    def __init__(self, rbf_kernel: IRBF, integral_bounds: Optional[List[Tuple[float, float]]],
                 measure: UniformMeasure, variable_names: str=''):
        """
        :param rbf_kernel: standard emukit rbf-kernel
        :param integral_bounds: defines the domain of the integral. List of D tuples, where D is the dimensionality
        of the integral and the tuples contain the lower and upper bounds of the integral
        i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]. None means infinite bounds.
        :param measure: A D-dimensional uniform measure
        :param variable_names: the (variable) name(s) of the integral
        """
        super().__init__(rbf_kernel=rbf_kernel, integral_bounds=integral_bounds, measure=measure,
                         variable_names=variable_names)

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
                raise ValueError("Upper bound of relevant integration domain must be larger than lower bound. Found a "
                                 "pair containing ({}, {}).".format(lb_d, ub_d))
        self._bounds_list_for_kernel_integrals = bounds
        self.reasonable_box_bounds = BoxBounds(name=variable_names, bounds=bounds)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        """
        RBF kernel with the first component integrated out aka kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x2, shape (1, N)
        """
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))
        kernel_mean = self.variance * (self.lengthscale * np.sqrt(np.pi / 2.) * (erf_up - erf_lo)).prod(axis=1)

        return kernel_mean.reshape(1, -1) * self.measure.density

    def qKq(self) -> float:
        """
        RBF kernel integrated over both arguments x1 and x2

        :returns: double integrated kernel
        """
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        prefac = self.variance * (2. * self.lengthscale**2)**self.input_dim
        diff_bounds_scaled = self._scaled_vector_diff(upper_bounds, lower_bounds)
        exp_term = np.exp(-diff_bounds_scaled**2) - 1.
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled * np.sqrt(np.pi)

        return float(prefac * (exp_term + erf_term).prod()) * self.measure.density ** 2

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in first argument) evaluated at x2
        :param x2: points at which to evaluate, shape (n_point N, input_dim)
        :return: the gradient with shape (input_dim, N)
        """
        lower_bounds = np.array([b[0] for b in self._bounds_list_for_kernel_integrals])
        upper_bounds = np.array([b[1] for b in self._bounds_list_for_kernel_integrals])
        exp_lo = np.exp(- self._scaled_vector_diff(x2, lower_bounds) ** 2)
        exp_up = np.exp(- self._scaled_vector_diff(x2, upper_bounds) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))

        fraction = ((exp_lo - exp_up) / (self.lengthscale * np.sqrt(np.pi / 2.) * (erf_up - erf_lo))).T

        return self.qK(x2) * fraction
