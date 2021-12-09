# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple

from ...quadrature.interfaces.base_gp import IBaseGaussianProcess
from .warped_bq_model import WarpedBayesianQuadratureModel
from .warpings import IdentityWarping


class VanillaBayesianQuadrature(WarpedBayesianQuadratureModel):
    """Vanilla Bayesian quadrature.

    Vanilla Bayesian quadrature uses a Gaussian process as surrogate for the integrand.
    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray):
        """
        :param base_gp: The underlying Gaussian process model.
        :param X: The initial locations of integrand evaluations, shape (n_points, input_dim).
        :param Y: The values of the integrand at X, shape (n_points, 1).
        """
        super(VanillaBayesianQuadrature, self).__init__(base_gp=base_gp, warping=IdentityWarping(), X=X, Y=Y)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute predictive means and variances of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim).
        :returns: Predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that
                  order all shapes (n_points, 1).
        """
        m, cov = self.base_gp.predict(X_pred)
        return m, cov, m, cov

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """Compute predictive means and covariance of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim).
        :returns: Predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
                  mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points).
        """
        m, cov = self.base_gp.predict_with_full_covariance(X_pred)
        return m, cov, m, cov

    def integrate(self) -> Tuple[float, float]:
        """Compute an estimator of the integral as well as its variance.

        :returns: Estimator of integral and its variance.
        """
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, 0]
        integral_var = self.base_gp.kern.qKq() - (kernel_mean_X @ self.base_gp.solve_linear(kernel_mean_X.T))[0, 0]
        return integral_mean, integral_var

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """Compute predictive gradients of mean and variance at given points.

        :param X: Points to compute gradients at, shape (n_points, input_dim).
        :returns: Tuple of gradients of mean and variance, shapes of both (n_points, input_dim).
        """
        return self.base_gp.get_prediction_gradients(X)
