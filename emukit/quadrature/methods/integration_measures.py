# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List


class IntegrationMeasure:

    def __init__(self, name: str):
        self.name = name


class GaussianMeasure(IntegrationMeasure):
    """The Gaussian measure"""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        super().__init__('GaussianMeasure')
        self.mean = mean
        self.covariance = covariance


class UniformMeasure(IntegrationMeasure):
    """The Uniform measure"""

    def __init__(self, bounds: List[Tuple[float, float]]):
        """
        :param bounds:
        """
        super().__init__('UniformMeasure')
        self.bounds = bounds
        self._check_bound_validity()
        self._compute_value()

    def _check_bound_validity(self) -> None:
        """checks if lower bounds are smaller than upper bounds."""
        for bounds_d in self.bounds:
            lb_d, ub_d = bounds_d
            if lb_d >= ub_d:
                raise ValueError("Upper bound of uniform measure must be larger than lower bound")

    def _compute_value(self) -> None:
        """computes density value"""
        differences = np.array([x[1] - x[0] for x in self.bounds])
        volume = np.prod(differences)

        # Todo: handle this better (it should not happen if upper bounds > lower bounds)
        # Todo: handle volume = 0 and return a zero integral?
        assert volume > 0
        self.density = 1./volume
