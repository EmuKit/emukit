# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, Union

from ...core.interfaces.models import IModel, IDifferentiable
from ...quadrature.interfaces.base_gp import IBaseGaussianProcess
from ...quadrature.kernels.bounds import BoxBounds
from ...quadrature.kernels.integration_measures import IntegrationMeasure
from .warpings import Warping


class WarpedBayesianQuadratureModel(IModel, IDifferentiable):
    """The general class for Bayesian quadrature (BQ) with a warped Gaussian process.

    Inference is performed with the warped GP, but the integral is computed on a Gaussian approximation.
    The warping of the base GP is encoded in the methods 'transform' and 'inverse_transform'

    Examples of warping-approximation pairs:
    - a moment matched squared GP (wsabi-m)
    - a linear approximation to a squared GP (wsabi-l)
    - no approximation if there is no warping (Vanilla BQ)
    - ...
    """
    def __init__(self, base_gp: IBaseGaussianProcess, warping: Warping, X: np.ndarray, Y: np.ndarray):
        """
        :param base_gp: The underlying Gaussian process model.
        :param warping: The warping of the underlying Gaussian process model.
        :param X: The initial locations of integrand evaluations, shape (n_points, input_dim).
        :param Y: The values of the integrand at X, shape (n_points, 1).
        """
        self._warping = warping
        self.base_gp = base_gp
        # set data to ensure that the base_gp get the correctly transformed observations.
        self.set_data(X, Y)

    @property
    def X(self) -> np.ndarray:
        return self.base_gp.X

    @property
    def Y(self) -> np.ndarray:
        return self._warping.transform(self.base_gp.Y)

    @property
    def integral_bounds(self) -> Union[None, BoxBounds]:
        """The integration bounds. ``None`` if integration domain is not bounded."""
        return self.base_gp.kern.integral_bounds

    @property
    def reasonable_box_bounds(self) -> BoxBounds:
        """Reasonable box bounds to search for observations. This box is used by the acquisition optimizer."""
        return self.base_gp.kern.reasonable_box_bounds

    @property
    def measure(self) -> Union[None, IntegrationMeasure]:
        """Probability measure used for integration. ``None`` for standard Lebesgue measure."""
        return self.base_gp.kern.measure

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from base-GP to integrand """
        return self._warping.transform(Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from integrand to base-GP """
        return self._warping.inverse_transform(Y)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute predictive means and variances of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim).
        :returns: Predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
                  all shapes (n_points, 1).
        """
        raise NotImplementedError

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """Compute predictive means and covariance of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :returns: Predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
                  mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points)
        """
        raise NotImplementedError

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute predictive means and covariance of warped GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :return: predictive mean, predictive full covariance of warped-GP, shapes (n_points, 1) and (n_points, n_points)
        """
        return self.predict_base_with_full_covariance(X_pred)[:2]

    def predict(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute predictive means and variances of warped-GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :return: predictive mean, predictive variances of warped-GP, both shapes (n_points, 1)
        """
        return self.predict_base(X_pred)[:2]

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Set the new data in the model.

        First, the model parameters that are not being optimized are updated, as they may depend on the new data,
        then new data is set in the model.

        :param X: Observation locations, shape (n_points, input_dim)
        :param Y: Integrand observations at X, shape (n_points, 1)
        """
        self._warping.update_parameters(**self.compute_warping_params(X, Y))
        self.base_gp.set_data(X, self._warping.inverse_transform(Y))

    def compute_warping_params(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Compute parameters of the warping that are dependent on data, and that are not being optimized.
        Override this method on case parameters are data dependent.

        :param X: Observation locations, shape (n_points, input_dim)
        :param Y: Integrand observations at X, shape (n_points, 1)
        :returns : Dictionary containing new warping parameters. Names of parameters are the keys.
        """
        return {}

    def optimize(self) -> None:
        """Optimizes the hyperparameters of the base GP"""
        self.base_gp.optimize()

    def integrate(self) -> Tuple[float, float]:
        """Compute an estimator of the integral as well as its variance.

        :returns: Estimator of integral and its variance.
        """
        raise NotImplementedError

    @staticmethod
    def symmetrize_matrix(A: np.ndarray) -> np.ndarray:
        """Symmetrize a matrix.

        The symmetrized matrix is computed as 0.5 (A + A.T).

        :param A: A square matrix, shape (N, N)
        :return: The symmetrized matrix, shape (N, N).
        """
        return 0.5 * (A + A.T)
