# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List


class IntegrationMeasure:
    """An integration measure"""

    def __init__(self, name: str):
        self.name = name


class GaussianMeasure(IntegrationMeasure):
    """The Gaussian measure"""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        """

        :param mean: the mean of the Gaussian
        :param covariance: the covariance matrix of the Gaussian
        """
        super().__init__('GaussianMeasure')
        self.mean = mean
        # Todo: check if we accept full covariance or just diagonal ones
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
        self._compute_density()

    def _check_bound_validity(self) -> None:
        """checks if lower bounds are smaller than upper bounds."""
        for bounds_d in self.bounds:
            lb_d, ub_d = bounds_d
            if lb_d >= ub_d:
                raise ValueError("Upper bound of uniform measure must be larger than lower bound. Found a pair "
                                 "containing (" + str(lb_d) + ", " + str(ub_d) + ").")

    def _compute_density(self) -> None:
        """computes density value"""
        differences = np.array([x[1] - x[0] for x in self.bounds])
        volume = np.prod(differences)

        if not volume > 0:
            raise NumericalPrecisionError("Domain volume of uniform measure is not positive. Its value is "
                                          + str(volume) + ". It might be numerical problems...")
        self.density = 1./volume


class NumericalPrecisionError(Exception):
    pass
