"""The Lebesgue measure."""

import numpy as np

from ...core.optimization.context_manager import ContextManager
from ..typing import BoundsType
from .domain import BoxDomain
from .integration_measure import IntegrationMeasure


class LebesgueMeasure(IntegrationMeasure):
    r"""The Lebesgue measure.

    The Lebesgue measure has density

    .. math::
        p(x)=\begin{cases} \hat{p} & x\in\text{domain}\\0 &\text{otherwise}\end{cases}.

    The value :math:`\hat{p} = 1` if the parameter ``normalized`` is set to ``False``
    and  :math:`\hat{p} = |\text{domain}|^{-1}` otherwise, where :math:`|\text{domain}|`
    is the volume (un-normalized Lebesgue measure) of the domain.

    :param domain: The Box domain.
    :param normalized: Weather the Lebesgue measure is normalized.

    """

    def __init__(self, domain: BoxDomain, normalized: bool = False):
        super().__init__(domain=domain, name="LebesgueMeasure")

        density = 1.0
        if normalized:
            differences = np.array([x[1] - x[0] for x in self.domain.bounds])
            volume = np.prod(differences)

            if volume <= 0:
                raise NumericalPrecisionError(
                    "Domain volume of uniform measure is not positive. Its value is {}.".format(volume)
                )
            density = float(1.0 / volume)

        self.density = density
        self.is_normalized = normalized
        self._input_dim = self.domain.dim

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
        return inside_upper_lower * self.density

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape)

    def reasonable_box(self) -> BoundsType:
        return self.domain.bounds

    def sample(self, num_samples: int, context_manager: ContextManager = None) -> np.ndarray:
        bounds = np.asarray(self.domain.bounds)

        samples = np.random.rand(num_samples, self.input_dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        if context_manager is not None:
            samples[:, context_manager.context_idxs] = context_manager.context_values

        return samples

    @classmethod
    def from_bounds(cls, bounds: BoundsType, normalized: bool = False):
        """Creates and instance of this class from integration bounds.

        :param bounds: List of d tuples :math:`[(a_1, b_1), (a_2, b_2), \dots, (a_d, b_d)]`,
                       where :math:`d` is the dimensionality of the domain and the tuple :math:`(a_i, b_i)`
                       contains the lower and upper bound of dimension :math:`i` defining the box domain.
        :param normalized: Weather the Lebesgue measure is normalized.
        :return: An instance of LebesgueMeasure.

        .. seealso::
            * :class:`emukit.quadrature.measures.BoxDomain`

        """
        domain = BoxDomain(name="", bounds=bounds)
        return cls(domain=domain, normalized=normalized)


class NumericalPrecisionError(Exception):
    pass
