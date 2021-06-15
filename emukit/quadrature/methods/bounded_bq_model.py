"""The bounded Bayesian quadrature model is with square-root warping."""

import numpy as np
from typing import Tuple

from ..interfaces.base_gp import IBaseGaussianProcess
from .warped_bq_model import WarpedBayesianQuadratureModel
from ..kernels.quadrature_rbf import QuadratureRBFIsoGaussMeasure
from .warpings import SquareRootWarping


class BoundedBQModel(WarpedBayesianQuadratureModel):
    """A warped Bayesian quadrature model that is upper bounded OR lower bounded by a constant.

    The integrand :math:`f(x)` is modeled as :math:`f(x) = f_* + 0.5 g(x)^2` for lower bounded functions, or
    :math:`f(x) = f^* - 0.5 g(x)^2` for upper bounded functions. The constants :math:`f_*` and :math:`f^*` are the
    lower and upper bound respectively, and :math:`g` is a Gaussian process (GP).

    The process :math:`f` induced by the Gaussian process :math:`g` is non-Gaussian. It is again approximated by a
    Gaussian process :math:`\\hat{f}` by linearizing :math:`f` around the mean of :math:`g`. The approximate GP
    :math:`\\hat{f}` is what is used by the predict-methods, and thus by the :meth:`integrate` method.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, bound: float,
                 is_lower_bounded: bool):
        """
        :param base_gp: The BaseGaussianProcess that models :math:`g`. Must use QuadratureRBFIsoGaussMeasure as kernel.
        :param X: The initial locations of integrand evaluations, shape (num_point, input_dim).
        :param Y: The values of the integrand at X, shape (num_points, 1).
        :param bound: the lower or upper bound :math:`\alpha`.
        :param is_lower_bounded: ``True``, if function is lower bounded. ``False`` if function is upper bounded.
        """
        # The integrate method is specific to QuadratureRBFIsoGaussMeasure, predict methods are only specific to
        # the approximation method used (Taylor expansion of the GP in this case).
        if not isinstance(base_gp.kern, QuadratureRBFIsoGaussMeasure):
            raise ValueError(f"{self.__class__.__name__} can only be used with QuadratureRBFIsoGaussMeasure kernel. "
                             f"Instead {type(base_gp.kern)} is given.")

        super(BoundedBQModel, self).__init__(base_gp=base_gp,
                                             warping=SquareRootWarping(offset=bound,
                                                                       inverted=not is_lower_bounded),
                                             X=X,
                                             Y=Y)

    @property
    def bound(self):
        """The bound :math:`\alpha` as defined in the model. The true bound of the integrand might be different."""
        return self._warping.offset

    @property
    def is_lower_bounded(self):
        """``True`` if the model is lower bounded, ``False`` if it is upper bounded."""
        return not self._warping.inverted

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the predictive mean and variance of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (num_points, input_dim).
        :returns: Predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that
                  order, all shapes (num_points, 1).
        """
        mean_base, var_base = self.base_gp.predict(X_pred)

        mean_approx = self.transform(mean_base)
        var_approx = var_base * (mean_base ** 2)
        return mean_approx, var_approx, mean_base, var_base

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """Compute predictive mean and covariance of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (num_points, input_dim)
        :returns: Predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
                  Mean shapes both (num_points, 1) and covariance shapes both (num_points, num_points)
        """
        mean_base, cov_base = self.base_gp.predict_with_full_covariance(X_pred)

        mean_approx = self.transform(mean_base)
        cov_approx = np.outer(mean_base, mean_base) * cov_base
        cov_approx = self._symmetrize(cov_approx)  # for numerical stability
        return mean_approx, cov_approx, mean_base, cov_base

    def integrate(self) -> Tuple[float, float]:
        """Compute the normal distribution of the integral value, in particular its mean estimator and variance.

        :returns: mean estimator of integral and its variance.
        """
        N, D = self.X.shape

        # weights and kernel
        X = self.X / np.sqrt(2)  # this is equivalent to scaling the lengthscale with sqrt(2)
        K = self.base_gp.kern.K(X, X)
        weights = self.base_gp.graminv_residual()
        W = np.outer(weights, weights)

        # kernel mean but with scaled lengthscale (multiplicative factor of 1/sqrt(2))
        X_means_vec = 0.5 * (self.X.T[:, :, None] + self.X.T[:, None, :]).reshape(D, -1).T
        qK = self.base_gp.kern.qK(X_means_vec, scale_factor=1./np.sqrt(2)).reshape(N, N)

        # integral mean
        integral_mean_second_term = 0.5 * np.sum(W * qK * K)
        if self.is_lower_bounded:
            integral_mean = self.bound + integral_mean_second_term
        else:
            integral_mean = self.bound - integral_mean_second_term

        # integral variance
        # The intgeral variance is not neede for the WSABI loop, hence it is not implemented yet
        # Will need to be implemented if the variance gets important.
        integral_variance = None
        return float(integral_mean), integral_variance

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """Compute model gradients of mean and variance at given points

        :param X: points to compute gradients at, shape (num_points, dim)
        :returns: Tuple of gradients of mean and variance, shapes of both (num_points, dim)
        """
        # gradient of mean
        mean_base, var_base = self.base_gp.predict(X)

        d_mean_dx = (self.base_gp.kern.dK_dx1(X, self.X) @ self.base_gp.graminv_residual())[:, :, 0].T
        # Todo: broadcasting?
        d_mean_dx = mean_base * d_mean_dx

        # gradient of variance
        dKdiag_dx = self.base_gp.kern.dKdiag_dx(X)
        dKxX_dx1 = self.base_gp.kern.dK_dx1(X, self.X)
        graminv_KXx = self.base_gp.solve_linear(self.base_gp.kern.K(self.base_gp.X, X))
        d_var_dx_base = dKdiag_dx - 2. * (dKxX_dx1 * np.transpose(graminv_KXx)).sum(axis=2, keepdims=False)

        d_var_dx = d_var_dx_base * mean_base ** 2 + 2 * var_base * mean_base * d_mean_dx

        if self.is_lower_bounded:
            return -d_mean_dx, d_var_dx
        return d_mean_dx, d_var_dx.T

    def update_parameters(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Update parameters.

        :param X: observation locations, shape (num_points, dim)
        :param Y: values of observations, shape (num_points, 1)
        """
        pass

    @staticmethod
    def _symmetrize(A: np.ndarray) -> np.ndarray:
        """Symmetrize a matrix.

        :param A: A square matrix, shape (N, N)
        :return: The symmetrized matrix 0.5 (A + A').
        """
        return 0.5 * (A + A.T)
