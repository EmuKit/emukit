# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List

from emukit.quadrature.interfaces.standard_kernels import IStandardKernel
from .integral_bounds import IntegralBounds


class QuadratureKernel:
    """
    Abstract class for covariance function of a Gaussian process than can be integrated

    Note that each specific implementation of this class must go with a specific standard kernel as input which
    inherits from IStandardKernel. This is because we both want the QuadratureKernel to be backend agnostic
    but at the same time QuadratureKernel needs access to specifics of the standard kernel.

    An example of a specific QuadratureKernel and IStandardKernel pair is QuadratureRBF and IRBF.
    """

    def __init__(self, kern: IStandardKernel, integral_bounds: List, integral_name: str='') -> None:
        """
        :param kern: standard emukit kernel
        :param integral_bounds: defines the domain of the integral. List of D tuples, where D is the dimensionality
        of the integral and the tuples contain the lower and upper bounds of the integral
        i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        :param integral_name: the (variable) name(s) of the integral
        """
        self.kern = kern
        self.integral_bounds = IntegralBounds(name=integral_name, bounds=integral_bounds)
        self.input_dim = self.integral_bounds.dim
        self.lower_bounds, self.upper_bounds = self.integral_bounds.get_lower_and_upper_bounds()

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        The kernel k(x1, x2) evaluated at x1 and x2

        :param x1: first argument of the kernel
        :param x2: second argument of the kernel
        :returns: kernel evaluated at x1, x2
        """
        return self.kern.K(x1, x2)

    # the following methods are integrals of a quadrature kernel
    def qK(self, x2: np.ndarray) -> np.ndarray:
        """
        Kernel with the first component integrated out aka. kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_points N, input_dim)
        :returns: kernel mean at location x2, shape (1, N)
        """
        raise NotImplementedError

    def Kq(self, x1: np.ndarray) -> np.ndarray:
        """
        Kernel with the second component integrated out aka. kernel mean

        :param x1: remaining argument of the once integrated kernel, shape (n_points N, input_dim)
        :returns: kernel mean at location x1, shape (N, 1)
        """
        raise NotImplementedError

    def qKq(self) -> np.float:
        """
        Kernel integrated over both arguments x1 and x2

        :returns: double integrated kernel
        """
        raise NotImplementedError

    # the following methods are gradients of a quadrature kernel
    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel wrt x1 evaluated at pair x1, x2

        :param x1: first argument of the kernel, shape = (n_points N, input_dim)
        :param x2: second argument of the kernel, shape = (n_points M, input_dim)
        :return: the gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M)
        """
        return self.kern.dK_dx1(x1, x2)

    def dK_dx2(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel wrt x2 evaluated at pair x1, x2
        Note that it is the transposed gradient wrt x1 evaluated at (x2, x1), i.e., the arguments are switched.

        :param x1: first argument of the kernel, shape = (n_points N, N, input_dim)
        :param x2: second argument of the kernel, shape = (n_points N, M, input_dim)
        :return: the gradient of the kernel wrt x2 evaluated at (x1, x2), shape (input_dim, N, M)
        """
        return np.transpose(self.dK_dx1(x1=x2, x2=x1), (0, 2, 1))

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """
        gradient of the diagonal of the kernel (the variance) v(x):=k(x, x) evaluated at x

        :param x: argument of the kernel, shape = (n_points M, input_dim)
        :return: the gradient of the diagonal of the kernel evaluated at x, shape (input_dim, M)
        """
        return self.kern.dKdiag_dx(x)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in first argument) evaluated at x2

        :param x2: N points at which to evaluate, shape = (n_points N, N, input_dim)
        :return: the gradient with shape (input_dim, N)
        """
        raise NotImplementedError

    def dKq_dx(self, x1: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in second argument) evaluated at x1
        :param x1: N points at which to evaluate, shape = (n_points N, N, input_dim)
        :return: the gradient with shape (N, input_dim)
        """
        raise NotImplementedError
