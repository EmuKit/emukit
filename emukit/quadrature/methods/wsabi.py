# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack
from typing import Tuple

from ...quadrature.interfaces.base_gp import IBaseGaussianProcess
from .warped_bq_model import WarpedBayesianQuadratureModel


class WSABI(WarpedBayesianQuadratureModel):
    """
    Base class for WSABI (Warped Sequential Active Bayesian Integration),

    Gunter et al. 2014
    Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature
    Advances in Neural Information Processing Systems (NeurIPS), 27, pp. 2789–2797.

    WSABI must be used with the RBF kernel and the Gaussian integration measure. This means that the kernel of base_gp
    must be of type QuadratureRBFIsoGaussMeasure.
    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, adapt_offset: bool=True):
        """
        :param base_gp: a model derived from BaseGaussianProcess. Must use QuadratureRBFIsoGaussMeasure as kernel.
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        :param adapt_offset: If True, offset of transformation will be adapted according to 0.8 x min(Y) as in
        Gunter et al.. If False the offset will be fixed to zero. Default is True.
        the offset will bet set to zero.
        """
        self.adapt_offset = adapt_offset
        if adapt_offset:
            self.offset = self._compute_offset()
        else:
            self.offset = 0
        super(WSABI, self).__init__(base_gp=base_gp, X=X, Y=Y)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from base-GP to integrand """
        return 0.5 * (Y * Y) + self.offset

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from integrand to base-GP """
        return np.sqrt(np.absolute(2.*(Y - self.offset)))

    def _compute_offset(self):
        """the value for the offset is given in Gunter et al. 2014 on page 3 in the footnote"""
        offset = 0.8 * min(self.Y)[0]
        return offset

    def _compute_and_set_offset_and_reset_data(self):
        """computes offset and sets new offset. Then resets data with new transform"""
        if self.adapt_offset:
            self.offset = self._compute_offset()

            # need to reset data in base GP because the transformation changed with the offset
            self.set_data(self.X, self.Y)
            # Todo: do we also need to refit the parameters here?

    @staticmethod
    def _symmetrize(A: np.ndarray) -> np.ndarray:
        """
        :param A: a square matrix, shape (N, N)
        :return: the symmetrized matrix 0.5 (A + A')
        """
        return 0.5 * (A + A.T)


class WSABIL(WSABI):
    """
     WSABI-L (Warped Sequential Active Bayesian Integration with linear approximation)

    Gunter et al. 2014
    Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature
    Advances in Neural Information Processing Systems (NeurIPS), 27, pp. 2789–2797.

    WSABI must be used with the RBF kernel and the Gaussian integration measure.

    The linear approximation is described in Gunter et al. in section 3.1, equations 9 and 10.
    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, adapt_offset: bool=True):
        """
        :param base_gp: a model derived from BaseGaussianProcess. Must use QuadratureRBFIsoGaussMeasure as kernel.
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        :param adapt_offset: If True, offset of transformation will be adapted according to 0.8 x min(Y) as in
        Gunter et al.. If False the offset will be fixed to zero. Default is True.
        the offset will bet set to zero.
        """
        super(WSABIL, self).__init__(base_gp, X, Y, adapt_offset)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        mean_base, var_base = self.base_gp.predict(X_pred)

        mean_approx = self.offset + 0.5 * (mean_base ** 2)
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

        mean_approx = self.offset + 0.5 * (mean_base ** 2)
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
        integral_mean = self.offset + 0.5 * np.sum(np.outer(weights, weights) * qK * K)

        # integral variance
        qK_weights = np.dot(qK, weights)  # 1 x N
        # Todo: think about imposing a linear solver instead
        lower_chol = self.base_gp.gram_chol()
        gram_inv_qK_weights = lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, qK_weights.T, lower=1)[0]),
                                            lower=0)[0]

        second_term = np.dot(qK_weights, gram_inv_qK_weights)
        # Todo: not done yet
        integral_variance = second_term[0, 0]

        return float(integral_mean), integral_variance
