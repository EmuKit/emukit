# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from ...core.acquisition import Acquisition
from ..methods import VanillaBayesianQuadrature


class SquaredCorrelation(Acquisition):
    r"""The squared correlation between the integral value and integrand evaluations
    under a Gaussian process model.

    .. math::
        a(x) = \frac{(\int k_N(s, x)p(s)\mathrm{d}s)^2}{\mathfrak{v}_N v_N(x)}\in [0, 1]

    where :math:`\mathfrak{v}_N` is the current integral variance given :math:`N` observations,
    :math:`v_N(x)` is the predictive integral variance if point :math:`x` was added newly,
    and :math:`k_N(s, x)` is the posterior kernel function.

    .. note::
        The squared correlation is identical to the integral-variance-reduction acquisition up to a
        global normalizing constant under a standard Gaussian process model.

    .. seealso::
        :class:`emukit.quadrature.acquisitions.IntegralVarianceReduction`

    :param model: A vanilla Bayesian quadrature model.

    """

    def __init__(self, model: VanillaBayesianQuadrature):
        self.model = model

    def has_gradients(self) -> bool:
        """Whether acquisition value has analytical gradient calculation available."""
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the acquisition function at x.

        :param x: The locations where to evaluate, shape (n_points, input_dim) .
        :return: The acquisition values at x, shape (n_points, 1).
        """
        return self._evaluate(x)[0]

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.float, np.ndarray, np.ndarray]:
        """Evaluates the acquisition function at x.

        :param x: The locations where to evaluate, shape (n_points, input_dim) .
        :return: The acquisition values at x (n_points, 1),
                 the current integral variance (float),
                 the predictive variance + noise at x (n_points, 1),
                 the predictive covariance between integral and x (n_points, 1).
        """
        integral_current_var, y_predictive_var, predictive_cov = self._value_terms(x)
        squared_correlation = predictive_cov**2 / (integral_current_var * y_predictive_var)
        return squared_correlation, integral_current_var, y_predictive_var, predictive_cov

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the acquisition function and its gradient.

        :param x: The locations where to evaluate, shape (n_points, input_dim).
        :return: The acquisition values and corresponding gradients at x,
                 shapes (n_points, 1) and (n_points, input_dim)
        """
        # value
        squared_correlation, integral_current_var, y_predictive_var, predictive_cov = self._evaluate(x)

        # gradient
        d_y_predictive_var_dx, d_predictive_cov_dx = self._gradient_terms(x)
        first_term = 2.0 * predictive_cov * d_predictive_cov_dx
        second_term = (predictive_cov**2 / y_predictive_var) * d_y_predictive_var_dx
        normalization = integral_current_var * y_predictive_var
        squared_correlation_gradient = (first_term - second_term) / normalization

        return squared_correlation, squared_correlation_gradient

    def _value_terms(self, x: np.ndarray) -> Tuple[np.float, np.ndarray, np.ndarray]:
        """Computes the terms needed for the squared correlation.

        :param x: The locations where to evaluate, shape (n_points, input_dim).
        :return: The current integral variance (float),
                 predictive variance + noise at x (n_points, 1),
                 predictive covariance between integral and x (n_points, 1).
        """
        integral_current_var = self.model.integrate()[1]
        y_predictive_var = self.model.predict(x)[1] + self.model.base_gp.observation_noise_variance

        qKx = self.model.base_gp.kern.qK(x)
        qKX = self.model.base_gp.kern.qK(self.model.base_gp.X)

        graminv_KXx = self.model.base_gp.solve_linear(self.model.base_gp.kern.K(self.model.base_gp.X, x))
        predictive_cov = np.transpose(qKx - np.dot(qKX, graminv_KXx))
        return integral_current_var, y_predictive_var, predictive_cov

    def _gradient_terms(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the terms needed for the gradient of the squared correlation.

        :param x: The locations where to evaluate, shape (n_points, input_dim).
        :return: The gradient of y_predictive_var and predictive_cov wrt. x at x, shapes (n_points, input_dim).
        """
        # gradient of predictive variance of y
        d_y_predictive_var_dx = self.model.get_prediction_gradients(x)[1].T

        # gradient of predictive covariance between integral and (x, y)-pair
        dqKx_dx = np.transpose(self.model.base_gp.kern.dqK_dx(x))

        qKX = self.model.base_gp.kern.qK(self.model.base_gp.X)
        qKX_graminv = np.transpose(self.model.base_gp.solve_linear(qKX.T))  # (1, N)

        dKXx_dx2 = self.model.base_gp.kern.dK_dx2(self.model.X, x)
        d_predictive_cov_dx = dqKx_dx - np.dot(qKX_graminv, np.transpose(dKXx_dx2))[0, :, :]

        return np.transpose(d_y_predictive_var_dx), d_predictive_cov_dx
