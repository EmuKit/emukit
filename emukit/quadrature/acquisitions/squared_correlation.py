# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack

from emukit.core.acquisition import Acquisition
from emukit.quadrature.methods import VanillaBayesianQuadrature


class SquaredCorrelation(Acquisition):
    """
    This acquisition function is the correlation between the integral and the new point(s)

    For GP-models, this acquisition function is identical to the integral-variance-reduction acquisition!

    .. math::
        \rho^2(x) = \frac{(\int k_N(x_1, x)\mathrm{d}x_1)^2}{\mathfrac{v}_N v_N(x)}

    where :math:`\mathfrac{v}_N` is the current integral variance given N observations X, :math:`v_N(x)` is the
    predictive integral variance if point x was added newly, and :math:`k_N(x_1, x)` is the posterior kernel function.
    """

    def __init__(self, model: VanillaBayesianQuadrature):
        """
        :param model: The vanilla Bayesian quadrature model
        """
        self.model = model

    def has_gradients(self) -> bool:
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the acquisition function at x.

        :param x: location where to evaluate (M, input_dim)
        :return: the acquisition function value at x
        """
        integral_current_var, integral_predictive_var, integral_predictive_cov = self._corr2_terms(x)
        return integral_predictive_cov**2 / (integral_current_var * integral_predictive_var)

    def _corr2_terms(self, x):
        """
        computes the terms needed for the squared correlation

        :param x: new candidate point x, contains the location and the fidelity to be evaluated
        :return: current integral variance, predictive variance + noise, once integrated predictive covariance
        """
        integral_current_var = self.model.integrate()[1]
        integral_predictive_var = self.model.predict(x)[1] + self.model.base_gp.observation_noise_variance

        qKx = self.model.base_gp.kern.qK(x)
        qKX = self.model.base_gp.kern.qK(self.model.base_gp.X)

        integral_predictive_cov = qKx - np.dot(qKX, self._Kinv_Kx(x))
        return integral_current_var, integral_predictive_var, integral_predictive_cov

    # following methods are for gradients
    def evaluate_with_gradients(self, x):
        """
        Evaluate the acquisition function with gradient

        :param x: location and fidelity
        :return: the acquisition function and its gradient evaluated at x
        """

        grad = self._corr2_gradient(x)
        return self.evaluate(x=x), grad

    def _corr2_gradient(self, x):
        """
        Computes the gradient of the acquisition function

        :param x: location at which to evaluate the gradient
        :return: the gradient at x
        """
        integral_current_var, integral_predictive_var, integral_predictive_cov = self._corr2_terms(x)
        d_predictive_var_dx, d_integral_predictive_cov_dx = self._corr2_gradient_terms(x)

        first_term = 2. * integral_predictive_cov * d_integral_predictive_cov_dx
        second_term = (1./integral_predictive_var) * d_predictive_var_dx * integral_predictive_cov**2
        normalization = integral_current_var * integral_predictive_var

        return (first_term - second_term) / normalization

    def _corr2_gradient_terms(self, x):
        """
        Computes the terms needed for the gradient of the squared correlation

        :param x: location at which to evaluate the gradient, contains fidelity levels
        :return: the gradient of (pred_var, int_pred_cov) at x
        """
        d_predictive_var_dx = -2. * (self.model.base_gp.kern.dK_dx(x, self.model.X) *
                                     self._Kinv_Kx(x).T).sum(axis=2, keepdims=True)

        d_integral_predictive_cov_dx = self.model.base_gp.kern.dqK_dx(x) \
                         - np.dot(self.model.base_gp.kern.dK_dx(x, self.model.X), self._Kinv_Kq())

        return d_predictive_var_dx, d_integral_predictive_cov_dx

    # helpers
    def _Kinv_Kx(self, x):
        """
        Inverse kernel Gram matrix multiplied with kernel function k(x, x') evaluated at existing
        datapoints x1=X and x2=x.

        .. math::
            K(X, X)^{-1} K (X, x)

        :param x: M locations at which to evaluate, shape ()
        :return: K(X,X)^-1 K(X, x) with shape (X.shape[0], M)
        """
        lower_chol = self.model.base_gp.gram_chol()
        KXx = self.model.base_gp.kern.K(self.model.base_gp.X, x)
        return lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, KXx, lower=1)[0]), lower=0)[0]

    def _Kinv_Kq(self):
        """
        Inverse kernel Gram matrix multiplied with kernel mean at self.models.X and high fidelity
        .. math::
            K(X, X)^{-1} \int K (X, x) dx

        :param x: N locations at which to evaluate
        :return: K(X,X)^-1 K(X, x) with shape (self.models.X.shape[0], N)
        """
        lower_chol = self.model.base_gp.gram_chol()
        qK = self.model.base_gp.kern.qK(x=self.model.base_gp.X)
        return lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, qK.T, lower=1)[0]), lower=0)[0]
