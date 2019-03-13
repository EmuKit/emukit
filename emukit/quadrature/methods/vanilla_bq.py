# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack
from typing import Tuple, Union

from .warped_bq_model import WarpedBayesianQuadratureModel
from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from .integration_measures import GaussianMeasure, UniformMeasure


class VanillaBayesianQuadrature(WarpedBayesianQuadratureModel):
    """
    class for vanilla Bayesian quadrature
    """

    def __init__(self, base_gp: IBaseGaussianProcess):
        """
        :param base_gp: a model derived from BaseGaussianProcess
        """
        super(VanillaBayesianQuadrature, self).__init__(base_gp)

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

    def integrate(self, measure: Union[GaussianMeasure, UniformMeasure] = None) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance.

        :param measure: The measure which is integrated against
        :returns: estimator of integral and its variance
        """
        if measure is None:
            integral_mean, integral_var = self._integrate_lebesgue()
        elif isinstance(measure, GaussianMeasure):
            integral_mean, integral_var = self._integrate_gaussian()
        elif isinstance(measure, UniformMeasure):
            integral_mean, integral_var = self._integrate_uniform()
        else:
            raise ValueError('unknown measure')
        return integral_mean, integral_var

    def _integrate_lebesgue(self):
        """computes integral against Lebesgue measure"""
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, 0]
        integral_var = self.base_gp.kern.qKq() - np.square(lapack.dtrtrs(self.base_gp.gram_chol(), kernel_mean_X.T,
                                                           lower=1)[0]).sum(axis=0, keepdims=True)[0][0]
        return integral_mean, integral_var

    def _integrate_gaussian(self):
        """computes integral against Gaussian measure"""
        # Todo: implement
        integral_mean, kernel_mean_X = self._compute_integral_mean_and_kernel_mean()
        integral_var = self.base_gp.kern.qKq() - np.square(lapack.dtrtrs(self.base_gp.gram_chol(), kernel_mean_X.T,
                                                           lower=1)[0]).sum(axis=0, keepdims=True)[0][0]
        return integral_mean, integral_var

    def _integrate_uniform(self):
        """computes integral against Uniform measure"""
        # Todo: implement
        integral_mean, kernel_mean_X = self._compute_integral_mean_and_kernel_mean()
        integral_var = self.base_gp.kern.qKq() - np.square(lapack.dtrtrs(self.base_gp.gram_chol(), kernel_mean_X.T,
                                                           lower=1)[0]).sum(axis=0, keepdims=True)[0][0]
        return integral_mean, integral_var

    def _compute_integral_mean_and_kernel_mean(self) -> Tuple[float, np.ndarray]:
        # Todo: remove this?
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, 0]
        return integral_mean, kernel_mean_X
