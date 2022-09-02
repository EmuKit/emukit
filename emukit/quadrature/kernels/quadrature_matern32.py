"""The product Matern32 kernel embeddings."""

from typing import Union

import numpy as np

from ...quadrature.interfaces.standard_kernels import IProductMatern32
from ..measures import IntegrationMeasure, LebesgueMeasure
from .quadrature_kernels import LebesgueEmbedding, QuadratureProductKernel


class QuadratureProductMatern32(QuadratureProductKernel):
    r"""Base class for a product Matern32 kernel augmented with integrability.

    The kernel is of the form :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{3}r_i ) e^{-\sqrt{3} r_i}.

    Above, :math:`d` is the input dimensionality, :math:`r_i =\frac{|x_i - x'_i|}{\lambda_i}`,
    is the scaled distance, :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i`
    is the :math:`i` th element of the ``lengthscales`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IProductMatern32`.
        Each subclass of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern32`
       * :class:`emukit.quadrature.kernels.QuadratureProductKernel`

    :param matern_kernel: The standard EmuKit product Matern32 kernel.
    :param measure: The integration measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        matern_kernel: IProductMatern32,
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


class QuadratureProductMatern32LebesgueMeasure(QuadratureProductMatern32, LebesgueEmbedding):
    """A product Matern32 kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern32`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern32`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param matern_kernel: The standard EmuKit product Matern32 kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern32, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(matern_kernel=matern_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        return {
            "domain": self.measure.domain.bounds[dim],
            "lengthscale": self.lengthscales[dim],
            "normalize": self.measure.is_normalized,
        }

    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        ell = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s3 = np.sqrt(3.0)
        first_term = 4.0 * ell / s3
        second_term = -np.exp(s3 * (x - b) / ell) * (b + 2.0 * ell / s3 - x)
        third_term = -np.exp(s3 * (a - x) / ell) * (x + 2.0 * ell / s3 - a)
        return (first_term + second_term + third_term) * normalization

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        ell = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        c = np.sqrt(3.0) * (b - a)
        qKq = 2.0 * ell / 3.0 * (2.0 * c - 3.0 * ell + np.exp(-c / ell) * (c + 3.0 * ell))
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        ell = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s3 = np.sqrt(3)
        exp_term_b = np.exp(s3 * (x - b) / ell)
        exp_term_a = np.exp(s3 * (a - x) / ell)
        first_term = exp_term_b * (-1 + (s3 / ell) * (x - b))
        second_term = exp_term_a * (+1 - (s3 / ell) * (a - x))
        return (first_term + second_term) * normalization
