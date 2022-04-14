# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from ...core.acquisition import Acquisition
from ..methods import VanillaBayesianQuadrature
from .squared_correlation import SquaredCorrelation


class MutualInformation(Acquisition):
    r"""The mutual information between the integral value and integrand evaluations
    under a Gaussian process model.

    The mutual information is a monotonic transformation of the squared correlation, hence it
    yields the same acquisition policy under a standard Gaussian process model.

    .. math::
        a(x) = -0.5 \log(1-\rho^2(x))

    where :math:`\rho^2` is the squared correlation.

    .. seealso::
        :class:`emukit.quadrature.acquisitions.SquaredCorrelation`

    :param model: A vanilla Bayesian quadrature model.
    """

    def __init__(self, model: VanillaBayesianQuadrature):
        self.rho2 = SquaredCorrelation(model)

    def has_gradients(self) -> bool:
        """Whether acquisition value has analytical gradient calculation available."""
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the acquisition function at x.

        :param x: The locations where to evaluate, shape (n_points, input_dim) .
        :return: The acquisition values at x, shape (n_points, 1).
        """
        rho2 = self.rho2.evaluate(x)
        mutual_information = -0.5 * np.log(1 - rho2)
        return mutual_information

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the acquisition function and its gradient.

        :param x: The locations where to evaluate, shape (n_points, input_dim) .
        :return: The acquisition values and corresponding gradients at x,
                 shapes (n_points, 1) and (n_points, input_dim)
        """
        # value
        mutual_information = self.evaluate(x)
        rho2, rho2_gradient = self.rho2.evaluate_with_gradients(x)

        # gradient
        mutual_information_gradient = (0.5 / (1 - rho2)) * rho2_gradient

        return mutual_information, mutual_information_gradient
