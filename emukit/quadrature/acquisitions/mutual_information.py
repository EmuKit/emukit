# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple

from ...core.acquisition import Acquisition
from .squared_correlation import SquaredCorrelation
from ..methods import VanillaBayesianQuadrature


class MutualInformation(Acquisition):
    """
    This acquisition function is the mutual information between the integral and the new point(s) under a GP-model.

    MutualInformation is a monotonic transform of SquaredCorrelation and hence yields the same acquisition policy in
    the case of vanilla-BQ.

    .. math::
        MI(x) = -0.5 \log(1-\rho^2(x))

    where :math:`\rho^2` is the SquaredCorrelation.
    """

    def __init__(self, model: VanillaBayesianQuadrature):
        """
        :param model: The vanilla Bayesian quadrature model
        """
        self.rho2 = SquaredCorrelation(model)

    def has_gradients(self) -> bool:
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the acquisition function at x.

        :param x: (n_points, input_dim) locations where to evaluate
        :return: (n_points, 1) the acquisition function value at x
        """
        rho2 = self.rho2.evaluate(x)
        mutual_information = -0.5 * np.log(1 - rho2)
        return mutual_information

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the acquisition function with gradient

        :param x: (n_points, input_dim) locations where to evaluate
        :return: acquisition value and corresponding gradient at x, shapes (n_points, 1) and (n_points, input_dim)
        """
        # value
        mutual_information = self.evaluate(x)
        rho2, rho2_gradient = self.rho2.evaluate_with_gradients(x)

        # gradient
        mutual_information_gradient = (0.5 / (1 - rho2)) * rho2_gradient

        return mutual_information, mutual_information_gradient
