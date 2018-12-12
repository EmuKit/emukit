# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple

from emukit.core.interfaces import IModel
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel


class IBaseGaussianProcess(IModel):
    """
    Interface for the quadrature base-GP model
    An instance of this can be passed as 'base_gp' to an ApproximateWarpedGPSurrogate object.

    If this GP is initialized with data, use the raw evaluations Y of the integrand and not transformed values.
    """

    def __init__(self, kern: QuadratureKernel) -> None:
        """
        If this GP is initialized with data X, Y, use the raw evaluations Y of the integrand and not transformed values
        as this is a general class that can be used with various quadrature methods. The transformation will be
        performed automatically when the quadrature method is initialized subsequently.
        :param kern: a quadrature kernel
        """
        self.kern = kern

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

        :param X_pred: points at which to predict, with shape (num_points, input_dim)
        :return: Predictive mean, predictive full co-variance shapes (num_points, num_points)
        """
        raise NotImplementedError

    def gram_chol(self) -> np.ndarray:
        """
        The lower triangular cholesky decomposition of the Gram matrix :math:`G(X, X) = K(X, X) + \sigma^2 I`.

        :return: a lower triangular cholesky of G(X, X)
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
