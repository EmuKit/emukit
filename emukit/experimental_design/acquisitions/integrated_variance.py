# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Contains integrated variance acquisition
"""
import numpy as np

from ...core.acquisition import Acquisition
from ...core.initial_designs import RandomDesign
from ...core import ParameterSpace

from ..interfaces import ICalculateVarianceReduction


class IntegratedVarianceReduction(Acquisition):
    """
    Acquisition function for integrated variance reduction
    """

    def __init__(self,
                 model: ICalculateVarianceReduction, space: ParameterSpace,
                 x_monte_carlo=None, num_monte_carlo_points: int=int(1e5)) -> None:
        """
        :param model: The emulation model
        :param space: The parameter space to select points within
        :param x_monte_carlo: The points to evaluate the GP at in order to estimate the integral of the variance.
                              Can be None and we will randomly sample within the constraints defined by the space.
        :param num_monte_carlo_points: Number of points to use to do Monte Carlo integration of variance.
                                       Not used if x_monte_carlo supplied.
        """

        self.model = model

        if x_monte_carlo is None:
            # Use RandomDesign to generate random points to do Monte Carlo integration
            # while respecting any constraints
            random_design = RandomDesign(space)
            self._x_monte_carlo = random_design.get_samples(num_monte_carlo_points)
        else:
            # Use user supplied points
            in_domain = space.check_points_in_domain(x_monte_carlo)
            if not np.all(in_domain):
                raise ValueError('Some or all of the points in x_monte_carlo are out of the valid domain.')
            self._x_monte_carlo = x_monte_carlo

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: The new training point(s) to evaluate with shape (n_points x 1)
        :return: A numpy array with shape (n_points x 1)
                 containing the values of the acquisition evaluated at each x row
        """

        n_eval_points = x.shape[0]
        integrated_variance = np.zeros((n_eval_points, 1))
        for i in range(n_eval_points):
            # Find variance reduction at each Monte Carlo point
            variance_reduction = self.model.calculate_variance_reduction(x[[i], :], self._x_monte_carlo)
            # Take mean to approximate integral per unit volume
            integrated_variance[i] = np.mean(variance_reduction)

        return integrated_variance

    @property
    def has_gradients(self):
        return False
