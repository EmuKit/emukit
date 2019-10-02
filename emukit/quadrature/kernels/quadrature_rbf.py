# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.special import erf
from typing import List, Tuple, Union

from .quadrature_kernels import QuadratureKernel
from ...quadrature.interfaces.standard_kernels import IRBF
from ...quadrature.kernels.integration_measures import IsotropicGaussianMeasure, IntegrationMeasure


class QuadratureRBF(QuadratureKernel):
    """
    Augments an RBF kernel with integrability

    Note 1: each standard kernel goes with a corresponding quadrature kernel, in this case the standard rbf kernel.
    Note 2: each child of this class implements a unique measure-integralBounds pair
    """

    def __init__(self, rbf_kernel: IRBF, integral_bounds: Union[None, List[Tuple[float, float]]],
                 measure: Union[None, IntegrationMeasure], variable_names: str= '') -> None:
        """
        :param rbf_kernel: standard emukit rbf-kernel
        :param integral_bounds: defines the domain of the integral. List of D tuples, where D is the dimensionality
        of the integral and the tuples contain the lower and upper bounds of the integral
        i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]. None for infinite bounds
        :param measure: an integration measure, None if there is no measure.
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
        RBF kernel with the first component integrated out aka. kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x2, shape (1, N)
        """
        raise NotImplementedError

    def Kq(self, x1: np.ndarray) -> np.ndarray:
        """
        RBF kernel with the second component integrated out aka. kernel mean

        :param x1: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x1, shape (N, 1)
        """
        return self.qK(x1).T

    def qKq(self) -> np.float:
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


class QuadratureRBFnoMeasure(QuadratureRBF):
    """
    And RBF kernel with integrability over no measure. Can only be used with finite integral bounds.

    Note that each standard kernel goes with a corresponding quadrature kernel, in this case QuadratureRBF
    """

    def __init__(self, rbf_kernel: IRBF, integral_bounds: List[Tuple[float, float]], variable_names: str= '') -> None:
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
        RBF kernel with the first component integrated out aka. kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x2, shape (1, N)
        """
        lower_bounds = self.integral_bounds.lower_bounds
        upper_bounds = self.integral_bounds.upper_bounds
        erf_lo = erf(self._scaled_vector_diff(lower_bounds, x2))
        erf_up = erf(self._scaled_vector_diff(upper_bounds, x2))
        kernel_mean = self.variance * (self.lengthscale * np.sqrt(np.pi / 2.) * (erf_up - erf_lo)).prod(axis=1)

        return kernel_mean.reshape(1, -1)

    def qKq(self) -> np.float:
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

        return np.float(prefac * (exp_term + erf_term).prod())

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

    Note that each standard kernel goes with a corresponding quadrature kernel, in this case QuadratureRBF
    """

    def __init__(self, rbf_kernel: IRBF, measure: IsotropicGaussianMeasure, variable_names: str= '') -> None:
        """
        :param rbf_kernel: standard emukit rbf-kernel
        :param measure: a Gaussian measure
        :param variable_names: the (variable) name(s) of the integral
        """
        super().__init__(rbf_kernel=rbf_kernel, integral_bounds=None, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        """
        RBF kernel with the first component integrated out aka. kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :returns: kernel mean at location x2, shape (1, N)
        """
        det_factor = (self.measure.variance / self.lengthscale ** 2 + 1) ** (self.input_dim / 2)
        scale = np.sqrt(self.lengthscale ** 2 + self.measure.variance)
        scaled_vector_diff = self._scaled_vector_diff(x2, self.measure.mean, scale)
        kernel_mean = (self.variance / det_factor) * np.exp(- np.sum(scaled_vector_diff ** 2, axis=1))
        return kernel_mean.reshape(1, -1)

    def qKq(self) -> np.float:
        """
        RBF kernel integrated over both arguments x1 and x2

        :returns: double integrated kernel
        """
        factor = (2 * self.measure.variance / self.lengthscale ** 2 + 1) ** (self.input_dim / 2)
        result = self.variance / factor
        return np.float(result)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in first argument) evaluated at x2

        :param x2: points at which to evaluate, shape (n_point N, input_dim)
        :return: the gradient with shape (input_dim, N)
        """
        qK_x = self.qK(x2)
        factor = 1. / (self.lengthscale ** 2 + self.measure.variance)
        return - (qK_x * factor) * (x2 - self.measure.mean).T
