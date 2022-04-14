"""The uniform measure."""


from typing import List, Tuple

import numpy as np

from emukit.core.optimization.context_manager import ContextManager

from .integration_measure import IntegrationMeasure


class UniformMeasure(IntegrationMeasure):
    r"""The Uniform measure.

    The uniform measure has density

    .. math::
        p(x)=\begin{cases} p & x\in\text{bounds}\\0 &\text{otherwise}\end{cases}.

    :param bounds: List of D tuples [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)], where D is
                   the input dimensionality and the tuple (lb_d, ub_d) contains the lower and upper bound
                   of the uniform measure in dimension d.

    """

    def __init__(self, bounds: List[Tuple[float, float]]):
        super().__init__("UniformMeasure")

        # checks if lower bounds are smaller than upper bounds.
        for (lb_d, ub_d) in bounds:
            if lb_d >= ub_d:
                raise ValueError(
                    "Upper bound of uniform measure must be larger than lower bound. Found a pair "
                    "containing ({}, {}).".format(lb_d, ub_d)
                )

        self.bounds = bounds
        # uniform measure has constant density which is computed here.
        self._density = self._compute_constant_density()

    def _compute_constant_density(self) -> float:
        differences = np.array([x[1] - x[0] for x in self.bounds])
        volume = np.prod(differences)

        if volume <= 0:
            raise NumericalPrecisionError(
                "Domain volume of uniform measure is not positive. Its value is {}.".format(volume)
            )
        return float(1.0 / volume)

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
        # Compute density: (i) check if points are inside the box. (ii) multiply this bool value with density.
        bounds_lower = np.array([b[0] for b in self.bounds])
        bounds_upper = np.array([b[1] for b in self.bounds])
        inside_lower = 1 - (x < bounds_lower)  # contains 1 if element in x is above its lower bound, 0 otherwise
        inside_upper = 1 - (x > bounds_upper)  # contains 1 if element in x is below its upper bound, 0 otherwise

        # Contains True if element in x is inside box, False otherwise.
        # The returned array thus contains self._density for each point inside the box and 0 otherwise.
        inside_upper_lower = (inside_lower * inside_upper).sum(axis=1) == x.shape[1]
        return inside_upper_lower * self._density

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the gradient of the density at x.

        :param x: Points at which the gradient is evaluated, shape (n_points, input_dim).
        :return: The gradient of the density at x, shape (n_points, input_dim).
        """
        return np.zeros(x.shape)

    def get_box(self) -> List[Tuple[float, float]]:
        """A meaningful box containing the measure.

        Outside this box, the measure should be zero or virtually zero.

        :return: The meaningful box.
        """
        return self.bounds

    def get_samples(self, num_samples: int, context_manager: ContextManager = None) -> np.ndarray:
        """Samples from the measure.

        :param num_samples: The number of samples to be taken.
        :param context_manager: The context manager that contains variables to fix and the values to fix them to.
                                If a context is given, this method samples from the conditional distribution.
        :return: The samples, shape (num_samples, input_dim).
        """

        D = len(self.bounds)
        bounds = np.asarray(self.bounds)

        samples = np.random.rand(num_samples, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        if context_manager is not None:
            samples[:, context_manager.context_idxs] = context_manager.context_values

        return samples


class NumericalPrecisionError(Exception):
    pass
