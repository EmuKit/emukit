# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


class IStandardKernel:
    """
    Interface for a standard kernel k(x, x') that in principle can be integrated
    Inherit from this class to construct wrappers for specific kernels e.g., the rbf
    """

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        The kernel k(x1, x2) evaluated at x1 and x2

        :param x1: first argument of the kernel
        :param x2: second argument of the kernel
        :returns: kernel evaluated at x1, x2
        """
        raise NotImplementedError

    # the following methods are gradients of the kernel
    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel wrt x1 evaluated at pair x1, x2

        :param x1: first argument of the kernel, shape = (n_points N, input_dim)
        :param x2: second argument of the kernel, shape = (n_points M, input_dim)
        :return: the gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M)
        """
        raise NotImplementedError

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """
        gradient of the diagonal of the kernel (the variance) v(x):=k(x, x) evaluated at x

        :param x: argument of the kernel, shape = (n_points M, input_dim)
        :return: the gradient of the diagonal of the kernel evaluated at x, shape (input_dim, M)
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
    def lengthscale(self) -> np.float:
        raise NotImplementedError

    @property
    def variance(self) -> np.float:
        raise NotImplementedError

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """
        gradient of the diagonal of the kernel v(x):=k(x, x) evaluated at x.

        :param x: argument of the kernel, shape = (n_points M, input_dim)
        :return: the gradient of the diagonal of the kernel evaluated at x, shape (input_dim, M)
        """
        raise np.zeros((x.shape[1], x.shape[0]))
