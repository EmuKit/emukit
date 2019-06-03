# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import scipy.stats
import numpy as np

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
        mean += self.jitter

        standard_deviation = np.sqrt(variance)
        y_minimum = np.min(self.model.Y, axis=0)
        cdf = scipy.stats.norm.cdf(y_minimum, mean, standard_deviation)
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

        mean += self.jitter
        u = (y_minimum - mean) / standard_deviation
        pdf = scipy.stats.norm.pdf(y_minimum, mean, standard_deviation)
        cdf = scipy.stats.norm.cdf(y_minimum, mean, standard_deviation)
        dcdf_dx = - pdf * (dmean_dx + dstandard_devidation_dx * u)

        return cdf, dcdf_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)

    
 class ProbabilityOfFeasibility(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: np.float64 = np.float64(0)) -> None:
        """
        This acquisition computes for a given input point the probability of satisfying the constraint
        C<0. For more information see:

        Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams,
        Bayesian Optimization with Unknown Constraints,
        https://arxiv.org/pdf/1403.5607.pdf

        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param jitter: Jitter to balance exploration / exploitation
        """
        self.model = model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the probability of of satisfying the constraint
        C<0.
        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        mean += self.jitter

        standard_deviation = np.sqrt(variance)
        cdf = scipy.stats.norm.cdf(0, mean, standard_deviation)
        return cdf

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the  probability of of satisfying the constraint
        C<0.
        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_devidation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u = - mean / standard_deviation
        pdf = scipy.stats.norm.pdf(0, mean, standard_deviation)
        cdf = scipy.stats.norm.cdf(0, mean, standard_deviation)
        dcdf_dx = - pdf * (dmean_dx + dstandard_devidation_dx * u)

        return cdf, dcdf_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)
