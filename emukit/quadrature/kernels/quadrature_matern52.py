"""The product Matern52 kernel embeddings."""

from typing import Optional, Union

import numpy as np

from ...quadrature.interfaces.standard_kernels import IProductMatern52
from ..measures import IntegrationMeasure
from ..typing import BoundsType
from .quadrature_kernels import QuadratureProductKernel


class QuadratureProductMatern52(QuadratureProductKernel):
    r"""A product Matern52 kernel augmented with integrability.

    The kernel is of the form :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{5} r_i + \frac{5}{3} r_i^2) \exp(- \sqrt{5} r_i).

    Above, :math:`d` is the input dimensionality, :math:`r_i =\frac{|x_i - x'_i|}{\lambda_i}`,
    is the scaled distance, :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i`
    is the :math:`i` th element of the ``lengthscales`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IProductMatern52`.
        Each subclass of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern52`
       * :class:`emukit.quadrature.kernels.QuadratureProductKernel`

    :param matern_kernel: The standard EmuKit product Matern52 kernel.
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param measure: The integration measure. ``None`` implies the standard Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        matern_kernel: IProductMatern52,
        integral_bounds: Optional[BoundsType],
        measure: Optional[IntegrationMeasure],
        variable_names: str = "",
    ) -> None:
        super().__init__(
            kern=matern_kernel, integral_bounds=integral_bounds, measure=measure, variable_names=variable_names
        )

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


class QuadratureProductMatern52LebesgueMeasure(QuadratureProductMatern52):
    """An product Matern52 kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern52`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern52`

    :param matern_kernel: The standard EmuKit product Matern52 kernel.
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern52, integral_bounds: BoundsType, variable_names: str = "") -> None:
        super().__init__(
            matern_kernel=matern_kernel, integral_bounds=integral_bounds, measure=None, variable_names=variable_names
        )

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z

    def _get_univariate_kwargs(self, dim: int) -> dict:
        return {"domain": self.integral_bounds.bounds[dim], "ell": self.lengthscales[dim]}

    def _qK_1d(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a, b = kwargs["domain"]
        ell = kwargs["ell"]
        s5 = np.sqrt(5)
        first_term = 16 * ell / (3 * s5)
        second_term = (
            -np.exp(s5 * (x - b) / ell) / (15 * ell) * (8 * s5 * ell**2 + 25 * ell * (b - x) + 5 * s5 * (b - x) ** 2)
        )
        third_term = (
            -np.exp(s5 * (a - x) / ell) / (15 * ell) * (8 * s5 * ell**2 + 25 * ell * (x - a) + 5 * s5 * (a - x) ** 2)
        )
        return first_term + second_term + third_term

    def _qKq_1d(self, **kwargs) -> float:
        a, b = kwargs["domain"]
        ell = kwargs["ell"]
        c = np.sqrt(5) * (b - a)
        bracket_term = 5 * a**2 - 10 * a * b + 5 * b**2 + 7 * c * ell + 15 * ell**2
        qKq = (2 * ell * (8 * c - 15 * ell) + 2 * np.exp(-c / ell) * bracket_term) / 15
        return float(qKq)

    def _dqK_dx_1d(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a, b = kwargs["domain"]
        ell = kwargs["ell"]
        s5 = np.sqrt(5)
        first_exp = -np.exp(s5 * (x - b) / ell) / (15 * ell)
        first_term = first_exp * (15 * ell - 15 * s5 * (x - b) + 25 / ell * (x - b) ** 2)
        second_exp = -np.exp(s5 * (a - x) / ell) / (15 * ell)
        second_term = second_exp * (-15 * ell + 15 * s5 * (a - x) - 25 / ell * (a - x) ** 2)
        return first_term + second_term
