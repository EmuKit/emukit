# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple

from emukit.core.interfaces import IModel
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBF
from emukit.quadrature.interfaces.standard_kernels import IStandardKernel, IRBF
from emukit.quadrature.kernels.integral_bounds import IntegralBounds


class IBaseGaussianProcess(IModel):
    """Interface for the quadrature base-GP model"""

    def __init__(self, standard_kern: IStandardKernel, integral_bounds: IntegralBounds) -> None:

        if isinstance(standard_kern, IRBF):
            self.kern = QuadratureRBF(rbf_kernel=standard_kern, integral_bounds=integral_bounds)
        else:
            raise NotImplementedError("Only RBF kernel is supported right now.")

    @property
    def observation_noise_variance(self) -> np.float:
        """
        Gaussian observation noise variance
        :return: The noise variance from some external GP model
        """
        raise NotImplementedError

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predictive mean and full co-variance at the locations X_pred

        :param X_pred: points at which to predict, with shape (number of points, dimension)
        :return: Predictive mean, predictive full co-variance
        """
        raise NotImplementedError

    def gram_chol(self) -> np.ndarray:
        """
        The lower triangular cholesky decomposition of the kernel Gram matrix

        :return: a lower triangular matrix being the cholesky matrix of the kernel Gram matrix
        """
        raise NotImplementedError

    def graminv_residual(self) -> np.ndarray:
        """
        The inverse Gram matrix multiplied with the mean-corrected data

        ..math::

            (K_{XX} + \sigma^2 I)^{-1} (Y - m(X))

        where the data is given by {X, Y} and m is the prior mean and sigma^2 the observation noise

        :return: the inverse Gram matrix multiplied with the mean-corrected data with shape: (number of datapoints, 1)
        """
        raise NotImplementedError

