# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack
from typing import Tuple

from ..interfaces.base_gp import IBaseGaussianProcess
from .warped_bq_model import WarpedBayesianQuadratureModel
from ..kernels.quadrature_rbf import QuadratureRBFIsoGaussMeasure


class BoundedSqrtTransformBayesianQuadratureModel(WarpedBayesianQuadratureModel):
    """
    A warped Bayesian quadrature model which is upper bounded OR lower bounded by a constant.

    The function is modeled as

    :math:`f(x) = \alpha + 0.5 g(x)^2` for lower bounded functions or
    :math:`f(x) = \alpha - 0.5 g(x)^2` for upper bounded functions.

    where :math:`g(x)' is modeled as a Gaussian process (base-gp) and :math:`\alpha' is a positive constant.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, bound: float, lower_bounded: bool):
        """
        :param base_gp: a model derived from BaseGaussianProcess. Must use QuadratureRBFIsoGaussMeasure as kernel.
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        :param bound: the constant lower or upper bound
        :param lower_bounded: True, if function will be lower bounded. False if function will be upper bounded.
        """
        self.lower_bounded = lower_bounded
        self.bound = bound
        super(BoundedSqrtTransformBayesianQuadratureModel, self).__init__(base_gp=base_gp, X=X, Y=Y)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from base-GP to integrand """
        if self.lower_bounded:
            return self.bound + 0.5 * (Y * Y)
        else:
            return self.bound - 0.5 * (Y * Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from integrand to base-GP """
        if self.lower_bounded:
            return np.sqrt(np.absolute(2. * (Y - self.bound)))
        else:
            return np.sqrt(np.absolute(2. * (self.bound - Y)))

    @staticmethod
    def _symmetrize(A: np.ndarray) -> np.ndarray:
        """
        :param A: a square matrix, shape (N, N)
        :return: the symmetrized matrix 0.5 (A + A')
        """
        return 0.5 * (A + A.T)


class BoundedBQSqrtTransformLinearApproxBQModel(BoundedSqrtTransformBayesianQuadratureModel):
    """
    A warped Bayesian quadrature model which is upper bounded OR lower bounded by a constant.

    The function is modeled as

    :math:`f(x) = \alpha + 0.5 g(x)^2` for lower bounded functions or
    :math:`f(x) = \alpha - 0.5 g(x)^2` for upper bounded functions.

    where :math:`g(x)' is modeled as a Gaussian process (base-gp) and :math:`\alpha' is a positive constant.

    The non-Gaussian process over :math:`f(x)' induced by the Gaussian process (GP) over :math:`g(x)' is then
    approximated by a Gaussian by linearizing :math:`f(x)' around the mean of :math:`g(x)'.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, bound: float, lower_bounded: bool):
        """
        :param base_gp: a model derived from BaseGaussianProcess. Must use QuadratureRBFIsoGaussMeasure as kernel.
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        :param bound: the constant lower or upper bound
        :param lower_bounded: True, if function will be lower bounded. False if function will be upper bounded.
        """

        # Todo: the integrate method is specific to QuadratureRBFIsoGaussMeasure, predict methods are only specific to
        #  the approximation method used.
        if not isinstance(base_gp.kern, QuadratureRBFIsoGaussMeasure):
            raise ValueError('BoundedSqrtTransformBayesianQuadratureModel can only be used with '
                             'QuadratureRBFIsoGaussMeasure kernel. Instead {} is given.'.format(type(base_gp.kern)))

        super(BoundedBQSqrtTransformLinearApproxBQModel, self).__init__(base_gp=base_gp, X=X, Y=Y, bound=bound,
                                                                        lower_bounded=lower_bounded)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        mean_base, var_base = self.base_gp.predict(X_pred)

        mean_approx = self.transform(mean_base)
        var_approx = var_base * (mean_base ** 2)

        return mean_approx, var_approx, mean_base, var_base

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """
        Computes predictive means and covariance of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :returns: predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
        mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points)
        """
        mean_base, cov_base = self.base_gp.predict_with_full_covariance(X_pred)

        mean_approx = self.transform(mean_base)
        cov_approx = np.outer(mean_base, mean_base) * cov_base
        cov_approx = self._symmetrize(cov_approx)  # for numerical stability

        return mean_approx, cov_approx, mean_base, cov_base

    def integrate(self) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance.

        :returns: estimator of integral and its variance
        """
        N, D = self.X.shape

        # weights and kernel
        X = self.X / np.sqrt(2)  # this is equivalent to scaling the lengthscale with sqrt(2)
        K = self.base_gp.kern.K(X, X)
        weights = self.base_gp.graminv_residual()

        # kernel mean but with scaled lengthscale (multiplicative factor of 1/sqrt(2))
        X_means_vec = 0.5 * (self.X.T[:, :, None] + self.X.T[:, None, :]).reshape(D, -1).T
        qK = self.base_gp.kern.qK(X_means_vec, scale_factor=1./np.sqrt(2)).reshape(N, N)

        # integral mean
        if self.lower_bounded:
            integral_mean = self.bound + 0.5 * np.sum(np.outer(weights, weights) * qK * K)
        else:
            integral_mean = self.bound - 0.5 * np.sum(np.outer(weights, weights) * qK * K)

        # integral variance
        # C1
        factor_1 = self.base_gp.kern.variance

        # C2
        qK_weights = np.dot(qK, weights)  # 1 x N
        # Todo: think about imposing a linear solver instead
        lower_chol = self.base_gp.gram_chol()
        gram_inv_qK_weights = lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, qK_weights.T, lower=1)[0]),
                                            lower=0)[0]

        second_term = np.dot(qK_weights, gram_inv_qK_weights)
        # Todo: not done yet
        integral_variance = second_term[0, 0]

        return float(integral_mean), integral_variance
