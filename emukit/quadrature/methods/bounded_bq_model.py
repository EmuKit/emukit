"""The bounded Bayesian quadrature model is with square-root warping."""

import numpy as np
from typing import Tuple, Optional

from ..interfaces.base_gp import IBaseGaussianProcess
from .warped_bq_model import WarpedBayesianQuadratureModel
from ..kernels.quadrature_rbf import QuadratureRBFIsoGaussMeasure
from .warpings import SquareRootWarping


class BoundedBayesianQuadrature(WarpedBayesianQuadratureModel):
    """A warped Bayesian quadrature model that is upper bounded OR lower bounded by a constant.

    The integrand :math:`f(x)` is modeled as :math:`f(x) = f_* + 0.5 g(x)^2` for lower bounded functions, or as
    :math:`f(x) = f^* - 0.5 g(x)^2` for upper bounded functions. The constants :math:`f_*` and :math:`f^*` are the
    lower and upper bound respectively, and :math:`g` is a Gaussian process (GP).

    The process :math:`f` induced by the Gaussian process :math:`g` is non-Gaussian. In order to obtain an analytic
    estimator for the integral value, the process :math:`f` is approximated by another Gaussian process
    :math:`\\hat{f}` by linearizing :math:`f` around the mean of :math:`g`. The approximate GP
    :math:`\\hat{f}` is implemented in the predict methods in this class, and it is also used by :math:`integrate`.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray,
                 lower_bound: Optional[float]=None, upper_bound: Optional[float]=None):
        """
        :param base_gp: The Gaussian process :math:`g`. Must use
               :class:`emukit.quadrature.kernels.QuadratureRBFIsoGaussMeasure` as kernel.
        :param X: The initial locations of integrand evaluations, shape (num_point, input_dim).
        :param Y: The values of the integrand at X, shape (num_points, 1).
        :param lower_bound: The lower bound  :math:`f_*` if the function is lower bounded.
        :param upper_bound: The upper bound :math:`f^*` if the function is lower bounded.
        """
        if lower_bound is None and upper_bound is None:
            raise ValueError("Either a lower or an upper bound needs to be given. Currently neither is given.")
        if lower_bound is not None and upper_bound is not None:
            raise ValueError("Either a lower or an upper bound needs to be given. Currently both are given.")

        bound = lower_bound
        is_lower_bounded = True

        if lower_bound is None:
            bound = upper_bound
            is_lower_bounded = False

        # The integrate method is specific to QuadratureRBFIsoGaussMeasure, predict methods are only specific to
        # the approximation method used (Taylor expansion of the GP in this case).
        if not isinstance(base_gp.kern, QuadratureRBFIsoGaussMeasure):
            raise ValueError(f"{self.__class__.__name__} can only be used with QuadratureRBFIsoGaussMeasure kernel. "
                             f"Instead {type(base_gp.kern)} is given.")

        super(BoundedBayesianQuadrature, self).__init__(base_gp=base_gp,
                                                        warping=SquareRootWarping(
                                                                 offset=bound, is_inverted=not is_lower_bounded
                                                             ),
                                                        X=X,
                                                        Y=Y)

    @property
    def bound(self):
        """The bound :math:`f^*` or :math:`f_*` as defined in the model."""
        return self._warping.offset

    @property
    def is_lower_bounded(self):
        """``True`` if the model is lower bounded, ``False`` if it is upper bounded."""
        return not self._warping.is_inverted

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the predictive mean and variance of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (num_points, input_dim).
        :returns: Predictive mean and variances of warped GP :math:`\\hat{f}`, and predictive mean and variances
                  of base-GP :math:`g` in that order, all shapes (num_points, 1).
        """
        mean_base, var_base = self.base_gp.predict(X_pred)

        mean_approx = self.transform(mean_base)
        var_approx = var_base * (mean_base ** 2)
        return mean_approx, var_approx, mean_base, var_base

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """Compute predictive mean and covariance of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (num_points, input_dim)
        :returns: Predictive mean and covariance of warped GP :math:`\\hat{f}`, predictive mean and covariance of
                  base-GP :math:`g` in that order. Mean shapes both (num_points, 1) and covariance shapes both
                  (num_points, num_points)
        """
        mean_base, cov_base = self.base_gp.predict_with_full_covariance(X_pred)

        mean_approx = self.transform(mean_base)
        cov_approx = np.outer(mean_base, mean_base) * cov_base
        cov_approx = self.symmetrize_matrix(cov_approx)  # force symmetry for numerical stability
        return mean_approx, cov_approx, mean_base, cov_base

    def integrate(self) -> Tuple[float, float]:
        """Compute the normal distribution of the integral value, in particular its mean estimator and variance.

        :returns: Mean estimator of integral and its variance. Currently, the variance returns None.
        """
        n_points, n_imput_dim = self.X.shape

        # weights and kernel
        X = self.X / np.sqrt(2)  # this is equivalent to scaling the lengthscale with sqrt(2)
        K = self.base_gp.kern.K(X, X)
        weights = self.base_gp.graminv_residual()
        Weights_outer = np.outer(weights, weights)

        # kernel mean but with scaled lengthscale (multiplicative factor of 1/sqrt(2))
        X_means_vec = 0.5 * (self.X.T[:, :, None] + self.X.T[:, None, :]).reshape(n_imput_dim, -1).T
        qK = self.base_gp.kern.qK(X_means_vec, scale_factor=1./np.sqrt(2)).reshape(n_points, n_points)

        # integral mean
        integral_mean_second_term = 0.5 * np.sum(Weights_outer * qK * K)
        if self.is_lower_bounded:
            integral_mean = self.bound + integral_mean_second_term
        else:
            integral_mean = self.bound - integral_mean_second_term

        # integral variance
        # The integral variance is not needed for the WSABI loop as WSABI uses uncertainty sampling.
        # For completeness, the integral variance will need to be implemented at a later point.
        integral_variance = None
        return float(integral_mean), integral_variance

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute model gradients of predictive mean and variance at given points.

        :param X: Points to compute gradients at, shape (num_points, input_dim)
        :returns: Tuple of gradients of mean and variance, shapes of both (num_points, input_dim)
        """
        # predictions and gradients of base model
        mean_base, var_base = self.base_gp.predict(X)
        d_mean_dx_base, d_var_dx_base = self.base_gp.get_prediction_gradients(X)

        # gradient of mean
        d_mean_dx = (self.base_gp.kern.dK_dx1(X, self.X) @ self.base_gp.graminv_residual())[:, :, 0].T
        d_mean_dx = mean_base * d_mean_dx  # broadcasting  (num_points, 1) and (num_points, num_dim)

        # gradient of variance
        d_var_dx = (mean_base ** 2) * d_var_dx_base + (2 * var_base * mean_base) * d_mean_dx_base  # broadcasting

        # the gradient of the mean of the lower bounded model is the negative gradient of the upper bounded model.
        if not self.is_lower_bounded:
            d_mean_dx *= -1.

        return d_mean_dx, d_var_dx
