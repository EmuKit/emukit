# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, Union

from ...core.acquisition import Acquisition
from ...core.interfaces import IDifferentiable
from ..methods import WarpedBayesianQuadratureModel


class UncertaintySampling(Acquisition):
    """Uncertainty sampling acquisition function for (warped) Bayesian quadrature.

    The acquisition function has the form :math:`a(x) = \var{x}` for the Lebesgue measure, and
    :math:`a(x) = \var(x)p(x) ^ q` for a measure with density :math:`p(x)`. The default value for the power :math:`q`
    is 2, but it can be set to a different value. :math:`\var(x)` is the posterior variance of the approximate
    Gaussian process (GP) on the integrand.
    """

    def __init__(self, model: Union[WarpedBayesianQuadratureModel, IDifferentiable], measure_power: float = 2):
        """
        :param model: A warped Bayesian quadrature model that has gradients.
        :param measure_power: The power of the measure. Default is 2. Only used if the measure is not the Lebesgue
               measure.
        """
        self.model = model
        self._measure_power = measure_power

    def has_gradients(self) -> bool:
        return True

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the predictive variances and the acquisition function.

        :param x: Locations at which to evaluate the acquisition function, shape (num_points x num_dim).
        :return: Values of the acquisition function at x and unweighted variances. Both shape (num_points x 1).
        """
        variances = self.model.predict(x)[1]
        if self.model.measure is None:
            return variances, variances
        else:
            weights = self.model.measure.compute_density(x).reshape(variances.shape)
            return variances * weights ** self._measure_power, variances

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the acquisition function.

        :param x: Locations at which to evaluate the acquisition function, shape (num_points x num_dim).
        :return: Values of the acquisition function at x, shape (num_points x 1).
        """
        return self._evaluate(x)[0]

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the acquisition function and compute its gradients.

        :param x: Locations at which to evaluate the acquisition function, shape (num_points x num_dim).
        :return: Values of the acquisition function at x, shape (num_points x 1),
                 and corresponding gradients, shape (num_points, num_dim).
        """
        p = self._measure_power
        variance_weighted, variance = self._evaluate(x)

        variance_gradient = self.model.get_prediction_gradients(x)[1]
        if self.model.measure is None:
            return variance, variance_gradient

        density = self.model.measure.compute_density(x)
        density_gradient = self.model.measure.compute_density_gradient(x)

        if p == 1:
            gradient_weighted = (density * variance_gradient.T).T + (variance[:, 0] * density_gradient.T).T
            return variance_weighted, gradient_weighted

        gradient_weighted = (density ** p * variance_gradient.T).T \
                            + (p * (variance[:, 0] * density ** (p - 1)) * density_gradient.T).T

        return variance_weighted, gradient_weighted
