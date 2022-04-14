"""The Gaussian measure."""

from typing import List, Tuple

import numpy as np

from emukit.core.optimization.context_manager import ContextManager

from .integration_measure import IntegrationMeasure


class IsotropicGaussianMeasure(IntegrationMeasure):
    r"""The isotropic Gaussian measure.

    The isotropic Gaussian measure has density

    .. math::
        p(x)=(2\pi\sigma^2)^{-\frac{D}{2}} e^{-\frac{1}{2}\frac{\|x-\mu\|^2}{\sigma^2}}

    where :math:`\mu` is the mean vector and :math:`\sigma^2` is the scalar variance
    parametrizing the measure.

    """

    def __init__(self, mean: np.ndarray, variance: float):
        """
        :param mean: The mean of the Gaussian measure, shape (num_dimensions, ).
        :param variance: The scalar variance of the Gaussian measure.
        """
        super().__init__("IsotropicGaussianMeasure")
        # check mean
        if not isinstance(mean, np.ndarray):
            raise TypeError("Mean must be of type numpy.ndarray, {} given.".format(type(mean)))

        if mean.ndim != 1:
            raise ValueError("Dimension of mean must be 1, dimension {} given.".format(mean.ndim))

        # check covariance
        if not isinstance(variance, float):
            raise TypeError("Variance must be of type float, {} given.".format(type(variance)))

        if not variance > 0:
            raise ValueError("Variance must be positive, current value is {}.".format(variance))

        self.mean = mean
        self.variance = variance
        self.num_dimensions = mean.shape[0]

    @property
    def full_covariance_matrix(self):
        """The full covariance matrix of the Gaussian measure."""
        return self.variance * np.eye(self.num_dimensions)

    @property
    def can_sample(self) -> bool:
        """Indicates whether the measure has sampling available.

        :return: ``True`` if sampling is available. ``False`` otherwise.
        """
        return True

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the density at x.

        :param x: Points at which density is evaluated, shape (num_points, input_dim).
        :return: The density at x, shape (num_points, ).
        """
        factor = (2 * np.pi * self.variance) ** (self.num_dimensions / 2)
        scaled_diff = (x - self.mean) / (np.sqrt(2 * self.variance))
        return np.exp(-np.sum(scaled_diff**2, axis=1)) / factor

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the gradient of the density at x.

        :param x: Points at which the gradient is evaluated, shape (num_points, input_dim).
        :return: The gradient of the density at x, shape (num_points, input_dim).
        """
        values = self.compute_density(x)
        return ((-values / self.variance) * (x - self.mean).T).T

    def get_box(self) -> List[Tuple[float, float]]:
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
        samples = self.mean + np.sqrt(self.variance) * np.random.randn(num_samples, self.num_dimensions)

        if context_manager is not None:
            # since the Gaussian is isotropic, fixing the value after sampling the joint is equal to sampling the
            # conditional.
            samples[:, context_manager.context_idxs] = context_manager.context_values

        return samples
