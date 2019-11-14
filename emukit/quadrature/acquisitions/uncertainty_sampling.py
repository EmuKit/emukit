# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, Union

from ...core.acquisition import Acquisition
from ...core.interfaces import IDifferentiable
from ..methods import WarpedBayesianQuadratureModel


class UncertaintySampling(Acquisition):
    """
    Uncertainty sampling acquisition function for (warped) Bayesian quadrature.

    The variance of the approximate transformed GP is used. If the integration measure is a probability measure,
    then the variance will be weighted with the probability density at each point.
    """

    def __init__(self, model: Union[WarpedBayesianQuadratureModel, IDifferentiable]):
        """
        :param model: A warped Bayesian quadrature model that has predictive gradients
        """
        self.model = model

    def has_gradients(self) -> bool:
        return True

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the acquisition function. If a probability measure is used, then the variance will be weighted by
        the density of the probability measure.

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values
        :return: (n_points x 1) array of acquisition function values, unweighted variances
        """
        variances = self.model.predict(x)[1]
        if self.model.measure is None:
            return variances, variances
        else:
            weights = self.model.measure.compute_density(x).reshape(variances.shape)
            return variances * weights, variances

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the acquisition function. If a probability measure is used, then the variance will be weighted by
        the density of the probability measure.

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values
        :return: (n_points x 1) array of acquisition function values
        """
        return self._evaluate(x)[0]

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional abstract method that must be implemented if has_gradients returns True.
        Evaluates value and gradient of acquisition function at x.

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values and gradient
        :return: Tuple contains an (n_points x 1) array of acquisition function values and (n_points x n_dims) array of
                 acquisition function gradients with respect to x
        """
        variance_weighted, variance = self._evaluate(x)

        variance_gradient = self.model.get_prediction_gradients(x)[1]
        if self.model.measure is None:
            return variance, variance_gradient
        else:
            density = self.model.measure.compute_density(x)
            density_gradient = self.model.measure.compute_density_gradient(x)
            gradient_weighted = (density * variance_gradient.T).T + (variance[:, 0] * density_gradient.T).T
            return variance_weighted, gradient_weighted
