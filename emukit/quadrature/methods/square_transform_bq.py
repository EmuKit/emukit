# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np

from .warped_bq_model import WarpedBayesianQuadratureModel
from ..interfaces.base_gp import IBaseGaussianProcess
from ..kernels.quadrature_rbf import QuadratureRBFIsoGaussMeasure


class LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(WarpedBayesianQuadratureModel):
    """
    A class for Bayesian quadrature over boundless Gaussian measure, with its
       RBF kernel Gaussian process warped with a square transform,
       and linearised for tractable integration.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray,
                 bound: float, alpha: float = None):
        """
        :param base_gp: the underlying GP model
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        :param bound: the upper bound
        :param alpha: the gap by which to step away from the bound; is important when
                the bound is zero so covariance isn't reduced to zero away from observed points.
                Will be re-set to min(old_alpha, 0.5 * (bound - max(Y)) ) every time
                new points are added to Y.
        """
        if not isinstance(base_gp.kern, QuadratureRBFIsoGaussMeasure):
            raise RuntimeError("Only RBF kernel and Gaussian measure supported")

        self.bound = bound  # assumed upper bound, and mean zero.
        # this is to ensure that the base_gp get the correct transform

        if alpha is None:
            alpha = 1 if bound == 0 else 0  # alpha will be set to
                                            # min(old_alpha, 0.5 * (bound - max(Y)) )
                                            # inside set(X, Y)

        self.alpha = alpha

        super().__init__(base_gp=base_gp, X=X, Y=Y)
        #TODO add a function that maps a non-zero mean and/or lower bound problem to a canonical zero-mean, upper-bound one.

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from base-GP to integrand.
        """

        res = self.bound - self.alpha - .5 * Y ** 2
        return res

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from integrand to base-GP.
        """
        res = np.sqrt( 2 * (self.bound - self.alpha - Y) )
        return res

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        m, cov = self.base_gp.predict(X_pred)
        m_lin, cov_lin =  self.transform(m), m ** 2 * cov  #TODO consider moving the derivative-calculation of transform into a method
        return m_lin, cov_lin, m, cov

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """
        Computes predictive means and covariance of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :returns: predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
        mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points)
        """
        m, cov = self.base_gp.predict_with_full_covariance(X_pred)
        m_lin, cov_lin =  self.transform(m), m.T * cov * m  #TODO consider moving the derivative-calculation of transform into a method
        return m_lin, cov_lin, m, cov

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        This method transforms the integrand y values and sets the data. If the alpha parameter
                is used, it will get updated if any new Y is between bound minus alpha and the bound.
        :param X: observed locations
        :param Y: observed integrand values
        """
        invalid_ind = np.where(Y >= self.bound)
        if invalid_ind[0].size != 0:
            raise ValueError("Some values are larger than the bound: %s " % ( Y[invalid_ind] ))

        if self.alpha > 0:
            self._update_alpha(Y)

        self.base_gp.set_data(X, self.inverse_transform(Y))

    def _update_alpha(self, Y: np.ndarray) -> None:
        """
        This method updates the stability parameter alpha if needed. .5 is hardcoded as that
                        parameter was reported to not affect performance in the WSABI paper.
        :param Y: observed integrand values
        """
        self.alpha = min(self.alpha, .5 * ( self.bound - np.max(Y) ) )

    def integrate(self) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance.
        :returns: estimator of integral and its variance
        """

        integral_mean = self._get_integral_mean()
        integral_var = self._get_integral_var_i1() - self._get_integral_var_i2()

        return integral_mean, integral_var

    def _get_integral_mean(self) -> float:
        """
        Computes an estimator of the integral
        :returns: estimator of integral
        """
        X = self.base_gp.X.squeeze()
        w = self.base_gp.graminv_residual().squeeze()
        lambd = np.array(self.base_gp.kern.lengthscale ** 2).squeeze()
        B = np.array(self.base_gp.kern.measure.variance).squeeze()
        b = np.array(self.base_gp.kern.measure.mean).squeeze()

        res = 0

        scale_1 = np.sqrt(2 * lambd)
        scale_2 = np.sqrt(2 * lambd + 4 * B)

        for n in range(w.shape[0]):
            for m in range(n):
                scaled_vector_diff_1 = self.base_gp.kern._scaled_vector_diff(X[n], X[m], scale_1)
                scaled_vector_diff_2 = self.base_gp.kern._scaled_vector_diff(X[n] + X[m], 2 * b, scale_2)

                res += w[n] * w[m] * np.exp(- np.sum(scaled_vector_diff_1 ** 2 + scaled_vector_diff_2 ** 2, axis=0))

            scaled_vector_diff_1 = self.base_gp.kern._scaled_vector_diff(X[n], X[n], scale_1)
            scaled_vector_diff_2 = self.base_gp.kern._scaled_vector_diff(2 * X[n], 2 * b, scale_2)

            res += .5 * w[n] ** 2 * np.exp(- np.sum(scaled_vector_diff_1 ** 2 + scaled_vector_diff_2 ** 2, axis=0))

        det_factor = (2 * B / lambd + 1) ** (self.input_dim / 2)

        res *= 2 * (self.base_gp.kern.variance ** 2 / det_factor)

        res = self.bound - .5 * res

        return res

    def _get_integral_var_i1(self) -> float:
        """
        Computes an estimator of the first integral that comprises the variance.
        :returns: estimator of integral and its variance

        math: \iint   k(x, X) G^{-1} y  k(x, x')   k(x', X) G^{-1} y  \pi (x) \pi (x') \text{d}x \text{d}x'
        """
        def _calc_for_nm(n, m):
            scaled_vector_diff_1 = self.base_gp.kern._scaled_vector_diff(X[n], X[m], scale_1)
            scaled_vector_diff_2 = self.base_gp.kern._scaled_vector_diff(X[m],
                                                                3 * b - 2 * X[n], scale_2)

            scaled_vector_diff_3 = self.base_gp.kern._scaled_vector_diff(np.dot(lamba_tilde / lambd, X[m]),
                                                                b - np.dot(lamba_tilde, \
                                                                    (2 * b - X[n]) / \
                                                                        (2 * lambd + 4 * B) + \
                                                                    X[n] / (2 * lambd)),
                                                                scale_3)
            return w[n] * w[m] * np.exp(- np.sum(scaled_vector_diff_1 ** 2 + \
                                                     scaled_vector_diff_2 ** 2 + \
                                                     scaled_vector_diff_3 ** 2, axis=0))


        X = self.base_gp.X.squeeze()
        w = self.base_gp.graminv_residual().squeeze()
        lambd = np.array(self.base_gp.kern.lengthscale ** 2).squeeze()
        B = np.array(self.base_gp.kern.measure.variance).squeeze()
        b = np.array(self.base_gp.kern.measure.mean).squeeze()

        lamba_tilde = np.array(lambd * (lambd + 2 * B) / (2 * lambd + 3 * B))

        res = 0

        scale_1 = np.sqrt(3 * lambd)
        scale_2 = np.sqrt(6 * lambd + 9 * B)
        scale_3 = np.sqrt(lamba_tilde + B)

        for n in range(w.shape[0]):
            for m in range(n):
                res += _calc_for_nm(n, m)
            res += .5 * _calc_for_nm(n, n)

        det_factor = ( (2 * B / lambd + 1) * (B / lamba_tilde + 1) ) ** (self.input_dim / 2)

        res *= 2 * (self.base_gp.kern.variance ** 3 / det_factor.squeeze())

        return res

    def _get_integral_var_i2(self) -> float:
        """
        Computes an estimator of the second integral that comprises the variance.
        :returns: estimator of integral and its variance

        math: \iint   k(x, X) G^{-1} y k(x, X) G^{-1} k(X, x')  k(x', X) G^{-1} y  \pi (x) \pi (x') \text{d}x \text{d}x'
        """
        #TODO kl = lk, and nm=mn. So some of the loops can be cut

        X = self.base_gp.X.squeeze()
        w = self.base_gp.graminv_residual().squeeze()
        lambd = np.array(self.base_gp.kern.lengthscale ** 2).squeeze()
        B = np.array(self.base_gp.kern.measure.variance).squeeze()
        b = np.array(self.base_gp.kern.measure.mean).squeeze()
        G = self.base_gp.gpy_model.posterior.woodbury_inv.squeeze()

        res = 0

        scale_1 = np.sqrt(lambd + B)
        scale_2 = np.sqrt( (lambd * (lambd + 2 * B)) / (lambd + B))

        for k in range(w.shape[0]):
            scaled_vector_diff_11 = self.base_gp.kern._scaled_vector_diff(X[k], b, scale_1)
            rbf_11 = np.exp(- np.sum(scaled_vector_diff_11 ** 2 , axis=0))

            res_l = 0
            for l in range(w.shape[0]):
                scaled_vector_diff_12 = self.base_gp.kern._scaled_vector_diff(X[l], b, scale_1)
                rbf_12 = np.exp(- np.sum(scaled_vector_diff_12 ** 2 , axis=0))

                res_m = 0
                for m in range(w.shape[0]):
                    scaled_vector_diff_21 = self.base_gp.kern._scaled_vector_diff(X[m],
                                               ( lambd * b + B * X[l]) / (lambd + B),
                                                                 scale_2)
                    res_m += w[m] * np.exp(- np.sum(scaled_vector_diff_21 ** 2 , axis=0))

                res_n = 0
                for n in range(w.shape[0]):
                    scaled_vector_diff_22 = self.base_gp.kern._scaled_vector_diff(X[n],
                                                ( lambd * b + B * X[k]) / (lambd + B),
                                                                 scale_2)
                    res_n += w[n] * res_m * np.exp(- np.sum(scaled_vector_diff_22 ** 2 , axis=0))

                res_l += G[k, l] * res_n * rbf_12

            res += res_l * rbf_11

        det_factor = (2 * B / lambd + 1) ** self.input_dim
        res *= (self.base_gp.kern.variance ** 4 / det_factor.squeeze())

        return res

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

        mean, var = self.base_gp.gpy_model.predict(X)

        return - mean * d_mean_dx, \
            d_mean_dx * var * mean + mean * d_var_dx.T * mean + mean * var * d_mean_dx

    @property
    def input_dim(self):
        return self.base_gp.kern.input_dim
