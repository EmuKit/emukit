# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List


class IntegrationMeasure:
    """An integration measure"""

    def __init__(self, name: str):
        """
        :param name: Name of the integration measure
        """
        self.name = name


class UniformMeasure(IntegrationMeasure):
    """The Uniform measure"""

    def __init__(self, bounds: List[Tuple[float, float]]):
        """
        :param bounds: List of D tuples, where D is the dimensionality of the domain and the tuples contain the lower
        and upper bounds of the box defining the uniform measure i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
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


class IsotropicGaussianMeasure(IntegrationMeasure):
    """The isotropic Gaussian measure"""

    def __init__(self, mean: np.ndarray, variance: float):
        """
        :param mean: the mean of the Gaussian
        :param variance: the variance of the isotropic covariance matrix of the Gaussian.
        """
        super().__init__('GaussianMeasure')
        self._check_input(mean, variance)
        self.mean = mean
        self.dim = mean.shape[0]
        self.variance = variance

    @property
    def covariance(self):
        return self._compute_full_covariance()

    def set_new_parameters(self, new_mean: np.ndarray = None, new_variance: float = None) -> None:
        """
        Set new mean and/or covariance. This also checks the input.
        :param new_mean: new mean of Gaussian
        :param new_variance: new variance of the isotropic covariance matrix of the Gaussian.
        """
        if new_mean is not None:
            # new mean, new var
            if new_variance is not None:
                self._check_input(new_mean, new_variance)
                self.mean = new_mean
                self.dim = new_mean.shape[0]
                self.variance = new_variance
            # new mean only
            else:
                self._check_input(new_mean, self.variance)
                self.mean = new_mean
                self.dim = new_mean.shape[0]
        else:
            # new var only
            if new_variance is not None:
                self._check_input(self.mean, new_variance)
                self.variance = new_variance

    def _check_input(self, mean: np.ndarray, variance: float) -> None:
        """
        checks type validity of mean and covariance inputs
        """
        # check mean
        if not isinstance(mean, np.ndarray):
            raise TypeError('Mean must be of type numpy.ndarray, ' + str(type(mean)) + ' given.')

        if mean.ndim != 1:
            raise ValueError('Dimension of mean must be 1, dimension ' + str(mean.ndim) + ' given.')

        # check covariance
        if not isinstance(variance, float):
            raise TypeError('Variance must be of type float, ' + str(type(variance)) + ' given.')

        if not variance > 0:
            raise ValueError('Variance must be positive, current value is ', variance, '.')

    def _compute_full_covariance(self):
        """Constructs full covariance matrix"""
        return self.variance * np.eye(self.dim)


class NumericalPrecisionError(Exception):
    pass
