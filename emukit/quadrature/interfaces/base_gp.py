# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple

from ...core.interfaces import IModel, IDifferentiable
from ..kernels.quadrature_kernels import QuadratureKernel


class IBaseGaussianProcess(IModel, IDifferentiable):
    """Interface for the quadrature base-GP model

    An instance of this can be passed as 'base_gp' to an
    :class:`emukit.quadrature.methods.warped_bq_model.WarpedBayesianQuadratureModel` instance.

    If this GP is initialized with data, use the raw evaluations Y of the integrand and not the transformed values.
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

    def solve_linear(self, z: np.ndarray) -> np.ndarray:
        """
        Solve the linear system G(X, X)x=z for x.
        G(X, X) is the Gram matrix :math:`G(X, X) = K(X, X) + \sigma^2 I`, of shape (num_dat, num_dat) and z is a
        matrix of shape (num_dat, num_obs).

        :param z: a matrix of shape (num_dat, num_obs)
        :return: the solution to the linear system G(X, X)x = z, shape (num_dat, num_obs)
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

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """Compute predictive gradients of mean and variance at given points.

        :param X: Points to compute gradients at, shape (n_points, input_dim).
        :returns: Tuple of gradients of mean and variance, shapes of both (n_points, input_dim).
        """
        # gradient of mean
        d_mean_dx = (self.kern.dK_dx1(X, self.X) @ self.graminv_residual())[:, :, 0].T

        # gradient of variance
        dKdiag_dx = self.kern.dKdiag_dx(X)
        dKxX_dx1 = self.kern.dK_dx1(X, self.X)
        graminv_KXx = self.solve_linear(self.kern.K(self.X, X))
        d_var_dx = dKdiag_dx - 2. * (dKxX_dx1 * np.transpose(graminv_KXx)).sum(axis=2, keepdims=False)

        return d_mean_dx, d_var_dx.T
