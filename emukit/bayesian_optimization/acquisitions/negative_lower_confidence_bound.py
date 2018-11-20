# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import numpy as np

from ...core.interfaces import IModel, IDifferentiable
from ...core.acquisition import Acquisition


class NegativeLowerConfidenceBound(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], beta: np.float64 = np.float64(1)) -> None:

        """
        This acquisition computes the negative lower confidence bound for a given input point. This is the same
        as optimizing the upper confidence bound if we would maximize instead of minimizing the objective function.
        For information as well as some theoretical insights see:

        Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
        Niranjan Srinivas, Andreas Krause, Sham Kakade, Matthias Seeger
        In Proceedings of the 27th International Conference  on  Machine  Learning

        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param beta: Is multiplied on the standard deviation to control exploration / exploitation
        """
        self.model = model
        self.beta = beta

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the negative lower confidence bound

        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        return - (mean - self.beta * standard_deviation)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the negative lower confidence bound and its derivative

        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        lcb = - (mean - self.beta * standard_deviation)

        dlcb_dx = - (dmean_dx - self.beta * dstandard_deviation_dx)

        return lcb, dlcb_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)
