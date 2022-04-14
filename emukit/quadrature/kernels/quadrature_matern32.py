"""The product Matern32 kernel embeddings."""

from typing import List, Optional, Tuple

import numpy as np

from emukit.quadrature.measures import IntegrationMeasure

from ...quadrature.interfaces.standard_kernels import IProductMatern32
from .quadrature_kernels import QuadratureKernel


class QuadratureProductMatern32(QuadratureKernel):
    """A product Matern32 kernel augmented with integrability.

    This class is compatible with the standard kernel :class:`IProductMatern32`.
    Each child of this class implements an embedding w.r.t. a specific integration measure.

    """

    """Augments a ProductMatern32 kernel with integrability."""

    def __init__(
        self,
        matern_kernel: IProductMatern32,
        integral_bounds: Optional[List[Tuple[float, float]]],
        measure: Optional[IntegrationMeasure],
        variable_names: str = "",
    ) -> None:
        """
        :param matern_kernel: The standard emukit product Matern32 kernel.
        :param integral_bounds: The integral bounds.
                                List of D tuples, where D is the dimensionality
                                of the integral and the tuples contain the lower and upper bounds of the integral
                                i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                                ``None`` if bounds are infinite.
        :param measure: The integration measure. ``None`` implies the standard Lebesgue measure.
        :param variable_names: The (variable) name(s) of the integral.
        """

        super().__init__(
            kern=matern_kernel, integral_bounds=integral_bounds, measure=measure, variable_names=variable_names
        )

    @property
    def nu(self):
        """The smoothness parameter of the product Matern32 kernel."""
        return 1.5

    @property
    def lengthscales(self):
        """The lengthscales of the product Matern32 kernel."""
        return self.kern.lengthscales

    @property
    def variance(self):
        """The scale of the product Matern32 kernel."""
        return self.kern.variance

    def qK(self, x2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def Kq(self, x1: np.ndarray) -> np.ndarray:
        return self.qK(x1).T

    def qKq(self) -> float:
        raise NotImplementedError

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dKq_dx(self, x1: np.ndarray) -> np.ndarray:
        return self.dqK_dx(x1).T


class QuadratureProductMatern32LebesgueMeasure(QuadratureProductMatern32):
    """An product Matern32 kernel augmented with integrability w.r.t. the standard Lebesgue measure."""

    def __init__(
        self, matern_kernel: IProductMatern32, integral_bounds: List[Tuple[float, float]], variable_names: str = ""
    ) -> None:
        """
        :param matern_kernel: The standard emukit product Matern32 kernel.
        :param integral_bounds: The integral bounds.
                                List of D tuples, where D is the dimensionality
                                of the integral and the tuples contain the lower and upper bounds of the integral
                                i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                                ``None`` if bounds are infinite.
        :param variable_names: The (variable) name(s) of the integral.
        """
        super().__init__(
            matern_kernel=matern_kernel, integral_bounds=integral_bounds, measure=None, variable_names=variable_names
        )

    def qK(self, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        if skip is None:
            skip = []

        qK = np.ones(x2.shape[0])
        for dim in range(x2.shape[1]):
            if dim in skip:
                continue
            qK *= self._qK_1d(x=x2[:, dim], domain=self.integral_bounds.bounds[dim], ell=self.lengthscales[dim])
        return qK[None, :] * self.variance

    def qKq(self) -> float:
        qKq = 1.0
        for dim in range(self.input_dim):
            qKq *= self._qKq_1d(domain=self.integral_bounds.bounds[dim], ell=self.lengthscales[dim])
        return self.variance * qKq

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        input_dim = x2.shape[1]
        dqK_dx = np.zeros([input_dim, x2.shape[0]])
        for dim in range(input_dim):
            grad_term = self._dqK_dx_1d(
                x=x2[:, dim], domain=self.integral_bounds.bounds[dim], ell=self.lengthscales[dim]
            )
            dqK_dx[dim, :] = grad_term * self.qK(x2, skip=[dim])[0, :]
        return dqK_dx

    # one dimensional integrals start here
    def _qK_1d(self, x: np.ndarray, domain: Tuple[float, float], ell: float) -> np.ndarray:
        """Unscaled kernel mean for 1D Matern32 kernel."""
        (a, b) = domain
        s3 = np.sqrt(3.0)
        first_term = 4.0 * ell / s3
        second_term = -np.exp(s3 * (x - b) / ell) * (b + 2.0 * ell / s3 - x)
        third_term = -np.exp(s3 * (a - x) / ell) * (x + 2.0 * ell / s3 - a)
        return first_term + second_term + third_term

    def _qKq_1d(self, domain: Tuple[float, float], ell: float) -> float:
        """Unscaled kernel variance for 1D Matern32 kernel."""
        a, b = domain
        r = b - a
        c = np.sqrt(3.0) * r
        qKq = 2.0 * ell / 3.0 * (2.0 * c - 3.0 * ell + np.exp(-c / ell) * (c + 3.0 * ell))
        return float(qKq)

    def _dqK_dx_1d(self, x, domain, ell):
        """Unscaled gradient of 1D Matern32 kernel mean."""
        s3 = np.sqrt(3)
        a, b = domain
        exp_term_b = np.exp(s3 * (x - b) / ell)
        exp_term_a = np.exp(s3 * (a - x) / ell)
        first_term = exp_term_b * (-1 + (s3 / ell) * (x - b))
        second_term = exp_term_a * (+1 - (s3 / ell) * (a - x))
        return first_term + second_term
