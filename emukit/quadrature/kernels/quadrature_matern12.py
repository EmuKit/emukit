"""The product Matern12 kernel embeddings."""

from typing import Union

import numpy as np

from ...quadrature.interfaces.standard_kernels import IProductMatern12
from ..measures import IntegrationMeasure, LebesgueMeasure
from .quadrature_kernels import LebesgueEmbedding, QuadratureProductKernel


class QuadratureProductMatern12(QuadratureProductKernel):
    r"""Base class for a product Matern12 (a.k.a. Exponential) kernel augmented with integrability.

    The kernel is of the form :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = e^{-r_i}.

    Above, :math:`d` is the input dimensionality, :math:`r_i =\frac{|x_i - x'_i|}{\lambda_i}`,
    is the scaled distance, :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i`
    is the :math:`i` th element of the ``lengthscales`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IProductMatern12`.
        Each subclass of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern12`
       * :class:`emukit.quadrature.kernels.QuadratureProductKernel`

    :param matern_kernel: The standard EmuKit product Matern12 kernel.
    :param measure: The integration measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        matern_kernel: IProductMatern12,
        measure: IntegrationMeasure,
        variable_names: str,
    ) -> None:
        super().__init__(kern=matern_kernel, measure=measure, variable_names=variable_names)

    @property
    def nu(self) -> float:
        """The smoothness parameter of the kernel."""
        return self.kern.nu

    @property
    def lengthscales(self) -> np.ndarray:
        r"""The lengthscales :math:`\lambda` of the kernel."""
        return self.kern.lengthscales

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        return self.kern.variance


class QuadratureProductMatern12LebesgueMeasure(QuadratureProductMatern12, LebesgueEmbedding):
    """A product Matern12 kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern12`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern12`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param matern_kernel: The standard EmuKit product Matern12 kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern12, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(matern_kernel=matern_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        return {
            "domain": self.measure.domain.bounds[dim],
            "ell": self.lengthscales[dim],
            "normalize": self.measure.is_normalized,
        }

    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        ell = parameters["ell"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        first_term = -np.exp((a - x) / ell)
        second_term = -np.exp((x - b) / ell)
        return normalization * ell * (2.0 + first_term + second_term)

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        ell = parameters["ell"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        qKq = 2.0 * ell * ((b - a) + ell * (np.exp(-(b - a) / ell) - 1.0))
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        ell = parameters["ell"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        first_term = np.exp((a - x) / ell)
        second_term = -np.exp((x - b) / ell)
        return (first_term + second_term) * normalization
