"""The Lebesgue measure."""


import numpy as np

from ...core.optimization.context_manager import ContextManager
from ..typing import BoundsType
from .integration_measure import IntegrationMeasure
from .domain import BoxDomain


# Todo: docstrint
class LebesgueMeasure(IntegrationMeasure):
    r"""The Lebesgue measure.

    The Lebesgue measure has density

    .. math::
        p(x)=\begin{cases} p & x\in\text{bounds}\\0 &\text{otherwise}\end{cases}.

    :param bounds: List of D tuples [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)], where D is
                   the input dimensionality and the tuple (lb_d, ub_d) contains the lower and upper bound
                   of the uniform measure in dimension d.

    """

    def __init__(self, bounds: BoundsType, normalized: bool=False):
        super().__init__(domain=BoxDomain(name="", bounds=bounds), name="LebesgueMeasure")

        self.is_normalized = normalized
        self._density = self._compute_constant_density()
        self._input_dim = self.domain.dim

    def _compute_constant_density(self) -> float:
        if self.is_normalized:
            return 1.0

        differences = np.array([x[1] - x[0] for x in self.domain.bounds])
        volume = np.prod(differences)

        if volume <= 0:
            raise NumericalPrecisionError(
                "Domain volume of uniform measure is not positive. Its value is {}.".format(volume)
            )
        return float(1.0 / volume)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def can_sample(self) -> bool:
        return True

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        # Compute density: (i) check if points are inside the box. (ii) multiply this bool value with density.
        bounds_lower = np.array([b[0] for b in self.domain.bounds])
        bounds_upper = np.array([b[1] for b in self.domain.bounds])
        inside_lower = 1 - (x < bounds_lower)  # contains 1 if element in x is above its lower bound, 0 otherwise
        inside_upper = 1 - (x > bounds_upper)  # contains 1 if element in x is below its upper bound, 0 otherwise

        # Contains True if element in x is inside box, False otherwise.
        # The returned array thus contains self._density for each point inside the box and 0 otherwise.
        inside_upper_lower = (inside_lower * inside_upper).sum(axis=1) == x.shape[1]
        return inside_upper_lower * self._density

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape)

    def get_box(self) -> BoundsType:
        return self.domain.bounds

    def get_samples(self, num_samples: int, context_manager: ContextManager = None) -> np.ndarray:
        D = len(self.domain.bounds)
        bounds = np.asarray(self.domain.bounds)

        samples = np.random.rand(num_samples, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        if context_manager is not None:
            samples[:, context_manager.context_idxs] = context_manager.context_values

        return samples


class NumericalPrecisionError(Exception):
    pass
