# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List

from emukit.core.interfaces.models import IModel
from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.kernels.integral_bounds import IntegralBounds


class WarpedBayesianQuadratureModel(IModel):
    """
    The general class for Bayesian quadrature (BQ) with a warped Gaussian process.

    Inference is performed with the warped GP, but the integral is computed on a Gaussian approximation.
    The warping of the base GP is encoded in the methods 'transform' and 'inverse_transform'

    Examples of warping-approximation pairs:
    - a moment matched squared GP (wsabi-m)
    - a linear approximation to a squared GP (wsabi-l)
    - no approximation if there is no warping (Vanilla BQ)
    - ...
    """
    def __init__(self, base_gp: IBaseGaussianProcess):
        """
        :param base_gp: the underlying GP model
        """
        self.base_gp = base_gp
        # this is to ensure that the base_gp get the correct transform
        self.set_data(base_gp.X, base_gp.Y)

    @property
    def X(self) -> np.ndarray:
        return self.base_gp.X

    @property
    def Y(self) -> np.ndarray:
        return self.transform(self.base_gp.Y)

    @ property
    def integral_bounds(self) -> IntegralBounds:
        return self.base_gp.kern.integral_bounds

    @property
    def integral_parameters(self) -> List:
        return self.base_gp.kern.integral_bounds.convert_to_list_of_continuous_parameters()

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from base-GP to integrand.
        """
        raise NotImplemented

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from integrand to base-GP.
        """
        raise NotImplemented

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        raise NotImplemented

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """
        Computes predictive means and covariance of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :returns: predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
        mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points)
        """
        raise NotImplemented

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes approximate predictive means and full co-variances of warped GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :return: predictive mean, predictive full covariance of warped-GP, shapes (n_points, 1) and (n_points, n_points)
        """
        return self.predict_base_with_full_covariance(X_pred)[:2]

    def predict(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of warped-GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :return: predictive mean, predictive variances of warped-GP, both shapes (n_points, 1)
        """
        return self.predict_base(X_pred)[:2]

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        This method transforms the integrand y values and sets the data
        :param X: observed locations
        :param Y: observed integrand values
        """
        self.base_gp.set_data(X, self.inverse_transform(Y))

    def optimize(self) -> None:
        """Optimizes the hyperparameters of the base GP"""
        self.base_gp.optimize()

    def integrate(self) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance.

        :returns: estimator of integral and its variance
        """
        raise NotImplemented
