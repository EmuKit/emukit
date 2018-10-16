# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import numpy as np

from GPyOpt.util.general import get_quantiles

from ...core.interfaces import IModel, IDifferentiable
from ...core.acquisition import Acquisition


class ProbabilityOfImprovement(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: np.float64 = np.float64(0)) -> None:
        """
        This acquisition computes for a given input point the probability of improving over the
        currently best observed function value. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param jitter: Jitter to balance exploration / exploitation
        """
        self.model = model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the probability of improving over the current best

        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        y_minimum = np.min(self.model.Y, axis=0)
        _, cdf, _ = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        return cdf

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the  probability of improving over the current best and its derivative

        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_devidation_dx = dvariance_dx / (2 * standard_deviation)

        y_minimum = np.min(self.model.Y, axis=0)
        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        dcdf_dx = - (pdf / standard_deviation) * (dmean_dx + dstandard_devidation_dx * u)

        return cdf, dcdf_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)
