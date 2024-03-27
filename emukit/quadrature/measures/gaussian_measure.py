# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""The Gaussian measure."""

from typing import Union

import numpy as np

from ...core.optimization.context_manager import ContextManager
from ..typing import BoundsType
from .integration_measure import IntegrationMeasure


class GaussianMeasure(IntegrationMeasure):
    r"""The Gaussian measure.

    The Gaussian measure has density

    .. math::
        p(x)=(2\pi)^{-\frac{d}{2}} \left(\prod_{j=1}^d \sigma_j^2\right)^{-\frac{1}{2}} e^{-\frac{1}{2}\sum_{i=1}^d\frac{(x_i-\mu_i)^2}{\sigma_i^2}}

    where :math:`\mu_i` is the :math:`i` th element of the ``mean`` parameter and
    :math:`\sigma_i^2` is :math:`i` th element of the ``variance`` parameter.

    :param mean: The mean of the Gaussian measure, shape (input_dim, ).
    :param variance: The variances of the Gaussian measure, shape (input_dim, ).
                     If a scalar value is given, all dimensions will have same variance.

    :raises TypeError: If ``mean`` is not of type :class:`ndarray`.
    :raises ValueError: If ``mean`` is not of dimension 1.
    :raises TypeError: If ``variance`` is neither of type :class:`float` nor of type :class:`ndarray`.
    :raises ValueError: If ``variance`` is of type :class:`float` but is non-positive.
    :raises ValueError: If ``variance`` is of type :class:`ndarray` but of other size than ``mean``.
    :raises ValueError: If ``variance`` is of type :class:`ndarray` and any of its elements is non-positive.

    """

    def __init__(self, mean: np.ndarray, variance: Union[float, np.ndarray]):
        super().__init__(domain=None, name="GaussianMeasure")
        # check mean
        if not isinstance(mean, np.ndarray):
            raise TypeError("Mean must be of type numpy.ndarray, {} given.".format(type(mean)))

        if mean.ndim != 1:
            raise ValueError("Dimension of mean must be 1, dimension {} given.".format(mean.ndim))

        # check covariance
        is_isotropic = False

        if isinstance(variance, float):
            if variance <= 0:
                raise ValueError("Variance must be positive, current value is {}.".format(variance))
            variance = np.full((mean.shape[0],), variance)
            is_isotropic = True

        elif isinstance(variance, np.ndarray):
            if variance.shape != mean.shape:
                raise ValueError(
                    "Variance has wrong shape; {} given but {} expected.".format(variance.shape, mean.shape)
                )
            if any(variance <= 0):
                raise ValueError(
                    "All elements of variance must be positive. At least one value seems to be non positive."
                )

        else:
            raise TypeError("Variance must be of type float or numpy.ndarray, {} given.".format(type(variance)))

        self.mean = mean
        self.variance = variance
        self._input_dim = mean.shape[0]
        self.is_isotropic = is_isotropic

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def full_covariance_matrix(self):
        """The full covariance matrix of the Gaussian measure."""
        return np.diag(self.variance)

    @property
    def can_sample(self) -> bool:
        return True

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        factor = (2 * np.pi) ** (self.input_dim / 2) * np.prod(np.sqrt(self.variance))
        scaled_diff = (x - self.mean) / (np.sqrt(2 * self.variance))
        return np.exp(-np.sum(scaled_diff**2, axis=1)) / factor

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        values = self.compute_density(x)
        diff = (x - self.mean) / self.variance
        return -diff * values[:, None]

    def reasonable_box(self) -> BoundsType:
        # The reasonable box is defined as the hypercube centered at the mean of the Gaussian with 10 standard
        # deviations expanding to either side (edge length of the cube are thus 20 standard deviations).
        # The factor 10 is somewhat arbitrary but well motivated as the Gaussian measure if virtually zero
        # outside of 10 standard deviations. See also the docstring of IntegrationMeasure.reasonable_box.
        factor = 10
        lower = self.mean - factor * np.sqrt(self.variance)
        upper = self.mean + factor * np.sqrt(self.variance)
        return list(zip(lower, upper))

    def sample(self, num_samples: int, context_manager: ContextManager = None) -> np.ndarray:
        samples = self.mean + np.sqrt(self.variance) * np.random.randn(num_samples, self.input_dim)

        if context_manager is not None:
            # Since the Gaussian is diagonal, fixing the value after sampling the joint is equal to sampling the
            # conditional.
            samples[:, context_manager.context_idxs] = context_manager.context_values

        return samples
