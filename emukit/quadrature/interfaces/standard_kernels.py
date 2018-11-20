# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


class IStandardKernel:
    """
    Interface for a standard kernel kernel that in principle can be integrated
    Inherit from this class to construct wrappers for specific kernels e.g., the rbf
    """

    def K(self, x: np.ndarray, x2: np.ndarray) -> np.float:
        """
        The kernel evaluated at x and x2

        :param x: the first argument of the kernel with shape (number of points N, input_dim)
        :param x2: the second argument of the kernel with shape (number of points M, input_dim)

        :return: the kernel matrix with shape (N, M)
        """
        raise NotImplementedError

    def dK_dx(self, x: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Gradient of the kernel. This method is required if a quadrature kernel with gradients is constructed

        :param x: first location at which to evaluate the kernel and component wrt which derivative has been taken
        :param x2: second location to evaluate

        :return: the gradient at x
        """
        raise NotImplementedError


class IRBF(IStandardKernel):
    """
    Interface for an RBF kernel
    Inherit from this class to wrap your standard rbf kernel.

    .. math::
        k(x, x') = \sigma^2 e^{-\frac{1}{2}\frac{\|x-x'\|^2}{\lambda^2}},

    where :math:`\sigma^2` is the `variance' property and :math:`\lambda` is the lengthscale property.
    """

    @property
    def lengthscale(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError
