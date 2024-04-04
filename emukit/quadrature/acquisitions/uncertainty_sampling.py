# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import numpy as np

from ...core.acquisition import Acquisition
from ...core.interfaces import IDifferentiable
from ..methods import WarpedBayesianQuadratureModel


class UncertaintySampling(Acquisition):
    r"""Uncertainty sampling.

    .. math::
        a(x) = \operatorname{var}(f(x)) p(x)^q

    where :math:`p(x)` is the density of the integration measure,
    :math:`\operatorname{var}(f(x))` is the predictive variance of the model at :math:`x`
    and :math:`q` is the ``measure_power`` parameter.

    :param model: A warped Bayesian quadrature model that has gradients.
    :param measure_power: The power :math:`q` of the measure. Default is 2.

    """

    def __init__(self, model: Union[WarpedBayesianQuadratureModel, IDifferentiable], measure_power: float = 2):
        self.model = model
        self._measure_power = measure_power

    @property
    def has_gradients(self) -> bool:
        """Whether acquisition value has analytical gradient calculation available."""
        return True

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the predictive variances and the acquisition function.

        :param x: The locations where to evaluate, shape (n_points, input_dim) .
        :return: The acquisition values at x and unweighted variances, both of shape (n_points, 1).
        """
        variances = self.model.predict(x)[1]
        weights = self.model.measure.compute_density(x).reshape(variances.shape)
        return variances * weights**self._measure_power, variances

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the acquisition function at x.

        :param x: The locations where to evaluate, shape (n_points, input_dim) .
        :return: The acquisition values at x, shape (n_points, 1).
        """
        return self._evaluate(x)[0]

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the acquisition function and its gradient.

        :param x: The locations where to evaluate, shape (n_points, input_dim).
        :return: The acquisition values and corresponding gradients at x,
                 shapes (n_points, 1) and (n_points, input_dim)
        """
        p = self._measure_power
        variance_weighted, variance = self._evaluate(x)

        variance_gradient = self.model.get_prediction_gradients(x)[1]
        density = self.model.measure.compute_density(x)
        density_gradient = self.model.measure.compute_density_gradient(x)

        if p == 1:
            gradient_weighted = (density * variance_gradient.T).T + (variance[:, 0] * density_gradient.T).T
            return variance_weighted, gradient_weighted

        gradient_weighted = (density**p * variance_gradient.T).T + (
            p * (variance[:, 0] * density ** (p - 1)) * density_gradient.T
        ).T

        return variance_weighted, gradient_weighted
