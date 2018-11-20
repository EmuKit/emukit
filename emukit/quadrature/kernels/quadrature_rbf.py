# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.special import erf

from .quadrature_kernels import QuadratureKernel
from emukit.quadrature.interfaces.standard_kernels import IRBF


class QuadratureRBF(QuadratureKernel):
    """
    Augments an RBF kernel with integrability

    Note that each standard kernel goes with a corresponding quadrature kernel, in this case QuadratureRBF
    """

    def __init__(self, rbf_kernel: IRBF, integral_bounds) -> None:
        super(QuadratureRBF, self).__init__(kern=rbf_kernel, integral_bounds=integral_bounds)

    @property
    def lengthscale(self):
        return self.kern.lengthscale

    @property
    def variance(self):
        return self.kern.variance

    # the following methods are integrals of a quadrature kernel
    def qK(self, x: np.ndarray) -> np.ndarray:
        """
        RBF kernel mean, i.e. the kernel integrated over the first argument as a function of its second argument

        :param x: N points at which kernel mean is evaluated, shape (N, input_dim)
        :returns: the kernel mean evaluated at N input points x, shape (1, N)
        """
        erf_lo = erf(self._scaled_vectordiff(self.lower_bounds, x))
        erf_up = erf(self._scaled_vectordiff(self.upper_bounds, x))
        kernel_mean = self.variance * (self.lengthscale * np.sqrt(np.pi/2.) * (erf_up - erf_lo)).prod(axis=1)

        return kernel_mean.reshape(1, -1)

    def Kq(self, x: np.ndarray) -> np.ndarray:
        """
        RBF transposed kernel mean,
        i.e. the kernel integrated over the second argument as a function of its first argument

        :param x: N points at which kernel mean is evaluated, shape (N, input_dim)
        :returns: the kernel mean evaluated at N input points x, shape (N,1)
        """
        return self.qK(x).T

    def qKq(self) -> np.float:
        """
        RBF kernel integrated over both parameters

        :returns: scalar value for the double integral of the kernel
        """

        prefac = self.variance * (2.* self.lengthscale**2)**self.input_dim
        diff_bounds_scaled = self._scaled_vectordiff(self.upper_bounds, self.lower_bounds)
        exp_term = np.exp(-diff_bounds_scaled**2) - 1.
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled * np.sqrt(np.pi)

        return np.float(prefac * (exp_term + erf_term).prod())

    # the following methods are gradients of a quadrature kernel
    def dqK_dx(self, x: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean evaluated at x
        :param x: N points at which to evaluate, shape = (N, input_dim)

        :return: the gradient with shape (input_dim, N)
        """

        exp_lo = np.exp(- self._scaled_vectordiff(x, self.lower_bounds)**2)
        exp_up = np.exp(- self._scaled_vectordiff(x, self.upper_bounds)**2)
        erf_lo = erf(self._scaled_vectordiff(self.lower_bounds, x))
        erf_up = erf(self._scaled_vectordiff(self.upper_bounds, x))

        fraction = ((exp_lo - exp_up)/(self.lengthscale * np.sqrt(np.pi/2.) * (erf_up - erf_lo))).T

        return self.qK(x) * fraction

    def dKq_dx(self, x: np.ndarray) -> np.ndarray:
        """
        gradient of the transposed kernel mean evaluated at x
        :param x: N points at which to evaluate, shape = (N, input_dim)

        :return: the gradient with shape (N, input_dim)
        """
        return self.dqK_dx(x).T

    # rbf-kernel specific helpers
    def _scaled_vectordiff(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Scaled element-wise vector difference between vectors v1 and v2

        .. math::
            \frac{v_1 - v_2}{\lambda \sqrt{2}}

        name mapping:
            \lambda: self.kern.lengthscale

        :param v1: first vector
        :param v2: second vector, must have same second dimensions as v1
        :return: scaled difference between v1 and v2, np.ndarray with unchanged dimensions

        """
        return (v1 - v2) / (self.lengthscale * np.sqrt(2))
