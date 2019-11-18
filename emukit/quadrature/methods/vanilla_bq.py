# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack
from typing import Tuple

from ...quadrature.interfaces.base_gp import IBaseGaussianProcess
from ...core.interfaces.models import IDifferentiable
from .warped_bq_model import WarpedBayesianQuadratureModel


class VanillaBayesianQuadrature(WarpedBayesianQuadratureModel, IDifferentiable):
    """
    Vanilla Bayesian quadrature.

    Vanilla Bayesian quadrature uses a Gaussian process as surrogate for the integrand.
    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray):
        """
        :param base_gp: a model derived from BaseGaussianProcess
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        """
        super(VanillaBayesianQuadrature, self).__init__(base_gp=base_gp, X=X, Y=Y)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from base-GP to integrand """
        return Y

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from integrand to base-GP """
        return Y

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        m, cov = self.base_gp.predict(X_pred)
        return m, cov, m, cov

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """
        Computes predictive means and covariance of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :returns: predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
        mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points)
        """
        m, cov = self.base_gp.predict_with_full_covariance(X_pred)
        return m, cov, m, cov

    def integrate(self) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance.

        :returns: estimator of integral and its variance
        """
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, 0]
        integral_var = self.base_gp.kern.qKq() - (kernel_mean_X @ self.base_gp.solve_linear(kernel_mean_X.T))[0, 0]
        return integral_mean, integral_var

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """
        Computes and returns model gradients of mean and variance at given points

        :param X: points to compute gradients at, shape (num_points, dim)
        :returns: Tuple of gradients of mean and variance, shapes of both (num_points, dim)
        """
        # gradient of mean
        d_mean_dx = (self.base_gp.kern.dK_dx1(X, self.X) @ self.base_gp.graminv_residual())[:, :, 0].T

        # gradient of variance
        dKdiag_dx = self.base_gp.kern.dKdiag_dx(X)
        dKxX_dx1 = self.base_gp.kern.dK_dx1(X, self.X)
        graminv_KXx = self.base_gp.solve_linear(self.base_gp.kern.K(self.base_gp.X, X))
        d_var_dx = dKdiag_dx - 2. * (dKxX_dx1 * np.transpose(graminv_KXx)).sum(axis=2, keepdims=False)

        return d_mean_dx, d_var_dx.T
