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
    :param variance: The variances of the Gaussian measure. If a scalar value is given, all dimensions
                     will have same variance.

    """

    def __init__(self, mean: np.ndarray, variance: Union[float, np.ndarray]):
        super().__init__("GaussianMeasure")
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
        self.input_dim = mean.shape[0]
        self.is_isotropic = is_isotropic

    @property
    def full_covariance_matrix(self):
        """The full covariance matrix of the Gaussian measure."""
        return np.diag(self.variance)

    @property
    def can_sample(self) -> bool:
        """Indicates whether the measure has sampling available.

        :return: ``True`` if sampling is available. ``False`` otherwise.
        """
        return True

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the density at x.

        :param x: Points at which density is evaluated, shape (n_points, input_dim).
        :return: The density at x, shape (n_points, ).
        """
        factor = (2 * np.pi) ** (self.input_dim / 2) * np.prod(np.sqrt(self.variance))
        scaled_diff = (x - self.mean) / (np.sqrt(2 * self.variance))
        return np.exp(-np.sum(scaled_diff**2, axis=1)) / factor

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the gradient of the density at x.

        :param x: Points at which the gradient is evaluated, shape (n_points, input_dim).
        :return: The gradient of the density at x, shape (n_points, input_dim).
        """
        values = self.compute_density(x)
        diff = (x - self.mean) / self.variance
        return -diff * values[:, None]

    def get_box(self) -> BoundsType:
        """A meaningful box containing the measure.

        Outside this box, the measure should be zero or virtually zero.

        :return: The meaningful box.
        """
        # Note: the factor 10 is somewhat arbitrary but well motivated. If this method is used to get a box for
        # data-collection, the box will be 2x 10 standard deviations wide in all directions, centered around the mean.
        # Outside the box the density is virtually zero.
        factor = 10
        lower = self.mean - factor * np.sqrt(self.variance)
        upper = self.mean + factor * np.sqrt(self.variance)
        return list(zip(lower, upper))

    def get_samples(self, num_samples: int, context_manager: ContextManager = None) -> np.ndarray:
        """Samples from the measure.

        :param num_samples: The number of samples to be taken.
        :param context_manager: The context manager that contains variables to fix and the values to fix them to.
                                If a context is given, this method samples from the conditional distribution.
        :return: The samples, shape (num_samples, input_dim).
        """
        samples = self.mean + np.sqrt(self.variance) * np.random.randn(num_samples, self.input_dim)

        if context_manager is not None:
            # Since the Gaussian is diagonal, fixing the value after sampling the joint is equal to sampling the
            # conditional.
            samples[:, context_manager.context_idxs] = context_manager.context_values

        return samples
