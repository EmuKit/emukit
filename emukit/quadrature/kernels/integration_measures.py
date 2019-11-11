# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List


class IntegrationMeasure:
    """An abstract class for a probability measure with a density"""

    def __init__(self, name: str):
        """
        :param name: Name of the integration measure
        """
        self.name = name

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the density at point x
        :param x: points at which density is computed, shape (num_points, dim)
        :return: the density at x, shape (num_points, )
        """
        raise NotImplementedError

    def get_box(self) -> List[Tuple[float, float]]:
        """
        Meaningful box-bounds around the measure. Outside this box, the measure should be virtually zero.

        :return: box in which the measure lies. List of D tuples, where D is the dimensionality and the tuples contain
        the lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        raise NotImplementedError

    @property
    def can_sample(self):
        """True if combined with IntegrationMeasureSampler"""
        raise NotImplementedError


class IntegrationMeasureSampler:
    """Augments the integration measure with sampling capabilities. This might be needed for some acquisition
    functions."""

    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        samples from the probability distribution defined by the integration measure.
        :param num_samples: number of samples
        :return: samples, shape (num_samples, dim)
        """
        raise NotImplementedError


class UniformMeasure(IntegrationMeasure, IntegrationMeasureSampler):
    """The Uniform measure"""

    def __init__(self, bounds: List[Tuple[float, float]]):
        """
        :param bounds: List of D tuples, where D is the dimensionality of the domain and the tuples contain the lower
        and upper bounds of the box defining the uniform measure i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        super().__init__('UniformMeasure')

        # checks if lower bounds are smaller than upper bounds.
        for (lb_d, ub_d) in bounds:
            if lb_d >= ub_d:
                raise ValueError("Upper bound of uniform measure must be larger than lower bound. Found a pair "
                                 "containing ({}, {}).".format(lb_d, ub_d))

        self.bounds = bounds
        # uniform measure has constant density which is computed here.
        self.density = self._compute_density()

    def _compute_density(self) -> float:
        differences = np.array([x[1] - x[0] for x in self.bounds])
        volume = np.prod(differences)

        if volume <= 0:
            raise NumericalPrecisionError("Domain volume of uniform measure is not positive. Its value is {}.".format(
                volume))
        return float(1. / volume)

    @property
    def can_sample(self):
        return True

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the density at point x
        :param x: points at which density is computed, shape (num_points, dim)
        :return: the density at x, shape (num_points, )
        """
        # compute density: (i) check if points are inside the box. (ii) multiply this bool value with density.
        bounds_lower = np.array([b[0] for b in self.bounds])
        bounds_upper = np.array([b[1] for b in self.bounds])
        inside_lower = 1 - (x < bounds_lower)  # contains 1 if element in x is above its lower bound, 0 otherwise
        inside_upper = 1 - (x > bounds_upper)  # contains 1 if element in x is below its upper bound, 0 otherwise
        # contain True if element in x is inside box, False otherwise. This array multiplied with the constant density
        # as done in the return statement yields self.density for a point inside the box and 0 otherwise.
        inside_upper_lower = (inside_lower * inside_upper).sum(axis=1) == x.shape[1]
        return inside_upper_lower * self.density

    def get_box(self) -> List[Tuple[float, float]]:
        """
        Meaningful box-bounds around the measure. Outside this box, the measure should be virtually zero.

        :return: box in which the measure lies. List of D tuples, where D is the dimensionality and the tuples contain
        the lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        return self.bounds

    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        samples from the uniform distribution.
        :param num_samples: number of samples
        :return: samples, shape (num_samples, dim)
        """
        D = len(self.bounds)
        samples = np.reshape(np.random.rand(num_samples * D), [num_samples, D])
        for d in range(D):
            samples[:, d] = samples[:, d] * (self.bounds[d][1] - self.bounds[d][0]) + self.bounds[d][0]
        return samples


class IsotropicGaussianMeasure(IntegrationMeasure, IntegrationMeasureSampler):
    """
    The isotropic Gaussian measure.

    An isotropic Gaussian is a Gaussian with scalar co-variance matrix. The density is
    :math:`p(x)=(2\pi\sigma^2)^{-\frac{D}{2}} e^{-\frac{1}{2}\frac{\|x-\mu\|^2}{\sigma^2}}`
    """

    def __init__(self, mean: np.ndarray, variance: float):
        """
        :param mean: the mean of the Gaussian, shape (dim, )
        :param variance: the scalar variance of the isotropic covariance matrix of the Gaussian.
        """
        super().__init__('GaussianMeasure')
        # check mean
        if not isinstance(mean, np.ndarray):
            raise TypeError('Mean must be of type numpy.ndarray, {} given.'.format(type(mean)))

        if mean.ndim != 1:
            raise ValueError('Dimension of mean must be 1, dimension {} given.'.format(mean.ndim))

        # check covariance
        if not isinstance(variance, float):
            raise TypeError('Variance must be of type float, {} given.'.format(type(variance)))

        if not variance > 0:
            raise ValueError('Variance must be positive, current value is {}.'.format(variance))

        self.mean = mean
        self.variance = variance
        self.dim = mean.shape[0]

    @property
    def full_covariance_matrix(self):
        return self.variance * np.eye(self.dim)

    @property
    def can_sample(self):
        return True

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the density at point x
        :param x: points at which density is computed, shape (num_points, dim)
        :return: the density at x, shape (num_points, )
        """
        factor = (2 * np.pi * self.variance) ** (self.dim / 2)
        scaled_diff = (x - self.mean) / (np.sqrt(2 * self.variance))
        return np.exp(- np.sum(scaled_diff ** 2, axis=1)) / factor

    def get_box(self) -> List[Tuple[float, float]]:
        """
        Meaningful box-bounds around the measure. Outside this box, the measure should be virtually zero.

        :return: box in which the measure lies. List of D tuples, where D is the dimensionality and the tuples contain
        the lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        # Note: the factor 10 is somewhat arbitrary but well motivated. If this method is used to get a box for
        # data-collection, the box will be 2x 10 standard deviations wide in all directions, centered around the mean.
        # Outside the box the density is virtually zero.
        factor = 10
        lower = self.mean - factor * np.sqrt(self.variance)
        upper = self.mean + factor * np.sqrt(self.variance)
        return list(zip(lower, upper))

    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        samples from the isotropic Gaussian distribution.
        :param num_samples: number of samples
        :return: samples, shape (num_samples, dim)
        """
        samples = np.reshape(np.random.randn(num_samples * self.dim), [num_samples, self.dim])
        return self.mean + np.sqrt(self.variance) * samples


class NumericalPrecisionError(Exception):
    pass
