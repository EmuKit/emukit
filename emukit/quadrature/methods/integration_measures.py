# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List, Union


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


class GaussianMeasure(IntegrationMeasure):
    """The Gaussian measure"""

    def __init__(self, mean: np.ndarray, covariance: Union[float, np.ndarray]):
        """
        :param mean: the mean of the Gaussian
        :param covariance: the covariance matrix of the Gaussian.
        """
        super().__init__('GaussianMeasure')
        self._cov_encoding = self._check_input(mean, covariance)
        self.mean = mean
        self.dim = mean.shape[0]
        self._covariance = covariance
        self._full_covariance = self._compute_full_covariance()

    @property
    def covariance(self):
        return self._full_covariance

    def set_new_parameters(self, mean: np.ndarray = None, covariance: Union[float, np.ndarray] = None) -> None:
        new_mean = mean
        new_covariance = covariance
        if new_mean is not None:
            # new mean, new cov
            if new_covariance is not None:
                new_cov_encoding = self._check_input(new_mean, new_covariance)

                self.mean = new_mean
                self.dim = new_mean.shape[0]
                self._cov_encoding = new_cov_encoding
                self._covariance = new_covariance
                self._full_covariance = self._compute_full_covariance()
            # new mean only
            else:
                self._check_input_mean(new_mean)
                self._check_match_mean_and_covariance(new_mean, self._full_covariance, 'full')
                self.mean = new_mean
        else:
            # new cov only
            if new_covariance is not None:
                new_cov_encoding = self._check_input_covariance(new_covariance)
                self._check_match_mean_and_covariance(self.mean, new_covariance, new_cov_encoding)

                self._cov_encoding = new_cov_encoding
                self._covariance = new_covariance
                self._full_covariance = self._compute_full_covariance()
            # no input given
            else:
                pass

    def _check_input(self, mean: np.ndarray, covariance: Union[float, np.ndarray]) -> str:
        self._check_input_mean(mean)
        cov_encoding = self._check_input_covariance(covariance)
        self._check_match_mean_and_covariance(mean, covariance, cov_encoding)
        return cov_encoding

    @staticmethod
    def _check_match_mean_and_covariance(mean: np.ndarray, covariance: Union[float, np.ndarray],
                                         cov_encoding: str):
        if (cov_encoding == 'full') or (cov_encoding == 'diag'):
            N1, N2 = covariance.shape
            if not mean.shape[0] == N1:
                raise ValueError('Dimensionality of mean and covariance must be equal. Given dimensions are '
                                 + str(mean.shape[0]) + ' and ' + str(N1) + '.')

    @staticmethod
    def _check_input_mean(mean: np.ndarray) -> None:
        """checks type and dimension of mean"""
        if not isinstance(mean, np.ndarray):
            raise TypeError('Mean must be of type numpy.ndarray, ' + str(type(mean)) + ' given.')
        if not mean.ndim == 1:
            raise ValueError('Dimension of mean must be 1, dimension ' + str(mean.ndim) + ' given.')

    @staticmethod
    def _check_input_covariance(covariance: Union[float, np.ndarray]) -> str:
        """checks type, dimension, and shape of covariance. Does not check sensible values."""
        # isotropic
        if isinstance(covariance, float):
            cov_encoding = 'iso'

        # diagonal or full
        elif isinstance(covariance, np.ndarray):
            if not covariance.ndim == 2:
                raise ValueError('Dimension of covariance must be 2. Dimension given is ' + str(covariance.ndim) + '.')

            N1, N2 = covariance.shape
            if N1 == N2:
                cov_encoding = 'full'
            elif N2 == 1:
                cov_encoding = 'diag'
            else:
                raise ValueError('Shape of covariance is (' + str(N1) + ',' + str(N2) + '). Expected is 2d array of '
                                 'shape (N, N) (full), or (N, 1) (diagonal), or a float (isotropic).')
        else:
            raise TypeError('Covariance must be of type float or numpy.ndarray, ' + str(type(covariance)) + ' given.')
        return cov_encoding

    def _compute_full_covariance(self):
        if self._cov_encoding == 'iso':
            full_cov = self._covariance * np.eye(self.dim)
        elif self._cov_encoding == 'diag':
            full_cov = np.diag(self._covariance[:, 0])
        elif self._cov_encoding == 'full':
            full_cov = self._covariance
        return full_cov


class NumericalPrecisionError(Exception):
    pass
