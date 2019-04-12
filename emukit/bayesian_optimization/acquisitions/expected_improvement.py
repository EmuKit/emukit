# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

from GPyOpt.util.general import get_quantiles
import numpy as np

from ...core.interfaces import IModel, IDifferentiable, IPriorHyperparameters
from ...core.acquisition import Acquisition


class ExpectedImprovement(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: np.float = np.float(0))-> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)
        dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)



class IntegratedExpectedImprovement(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable, IPriorHyperparameters], jitter: np.float64 = np.float64(0),
                 n_samples = 10) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. This function integrates over hyper-parameters the model by computing the  average of the
        expected improvements for all samples. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.jitter = jitter
        self.n_samples = n_samples
        self.samples = self.model.generate_hyperparameters_samples(n_samples)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the integrated Expected Improvement with respect to the hyper-parameters of the model. Averages the
        improvement for all the samples.

        :param x: points where the acquisition is evaluated.
        :return: numpy array with the integrated expected improvement at the points x.
        """

        if x.ndim == 1: x = x[None, :]
        improvement = 0

        for sample in self.samples:
            self.model.fix_model_hyperparameters(sample)
            acquisition = ExpectedImprovement(self.model, self.jitter)
            improvement += acquisition.evaluate(x)

        return improvement/self.n_samples

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative integrating over the hyper-parameters of the model

        :param x: locations where the evaluation with gradients is done.
        :return: tuple containing numpy arrays with the integrated expected improvement at the points x
        and its gradient.
        """

        if x.ndim == 1: x = x[None, :]
        improvement = 0
        dimprovement_dx = 0

        for sample in self.samples:
            self.model.fix_model_hyperparameters(sample)
            acquisition = ExpectedImprovement(self.model, self.jitter)
            improvement_sample, dimprovement_dx_sample = acquisition.evaluate_with_gradients(x)
            improvement += improvement_sample
            dimprovement_dx += dimprovement_dx_sample

        return improvement/self.n_samples, dimprovement_dx/self.n_samples

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)