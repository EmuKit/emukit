"""The Brownian motion kernel embeddings."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Tuple

import numpy as np

from ...quadrature.interfaces.standard_kernels import IBrownian, IProductBrownian
from ..kernels import QuadratureKernel
from ..measures import IntegrationMeasure
from ..typing import BoundsType


class QuadratureBrownian(QuadratureKernel):
    r"""A Brownian motion kernel augmented with integrability.

    .. math::
        k(x, x') = \sigma^2 \operatorname{min}(x, x')\quad\text{with}\quad x, x' \geq 0,

    where :math:`\sigma^2` is the ``variance`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IBrownian`.
        Each child of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureKernel`

    :param brownian_kernel: The standard EmuKit Brownian motion kernel.
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param measure: The integration measure. ``None`` implies the standard Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    :raises ValueError: If ``integral_bounds`` have wrong length.

    """

    def __init__(
        self,
        brownian_kernel: IBrownian,
        integral_bounds: Optional[BoundsType],
        measure: Optional[IntegrationMeasure],
        variable_names: str = "",
    ) -> None:

        if integral_bounds is not None:
            if len(integral_bounds) != 1:
                raise ValueError("Integral bounds for Brownian motion kernel must be 1-dimensional.")

        super().__init__(
            kern=brownian_kernel, integral_bounds=integral_bounds, measure=measure, variable_names=variable_names
        )

        lower_bounds_x = self.reasonable_box.lower_bounds[0, :]
        if any(lower_bounds_x < 0):
            raise ValueError(
                "The domain defined by the reasonable box seems to allow negative values. "
                "Brownian motion is only defined for positive input values."
            )

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
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


class QuadratureBrownianLebesgueMeasure(QuadratureBrownian):
    """A Brownian motion kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureBrownian`

    :param brownian_kernel: The standard EmuKit Brownian motion kernel.
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, brownian_kernel: IBrownian, integral_bounds: BoundsType, variable_names: str = "") -> None:
        super().__init__(
            brownian_kernel=brownian_kernel,
            integral_bounds=integral_bounds,
            measure=None,
            variable_names=variable_names,
        )

    def qK(self, x2: np.ndarray) -> np.ndarray:
        lb = self.integral_bounds.lower_bounds
        ub = self.integral_bounds.upper_bounds
        kernel_mean = ub * x2 - 0.5 * x2**2 - 0.5 * lb**2
        return self.variance * kernel_mean.T

    def qKq(self) -> float:
        lb = self.integral_bounds.lower_bounds[0, 0]
        ub = self.integral_bounds.upper_bounds[0, 0]
        qKq = 0.5 * ub * (ub**2 - lb**2) - (ub**3 - lb**3) / 6 - 0.5 * lb**2 * (ub - lb)
        return float(self.variance * qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        ub = self.integral_bounds.upper_bounds
        return self.variance * (ub - x2).T


class QuadratureProductBrownian(QuadratureKernel):
    r"""A product Brownian kernel augmented with integrability.

    The kernel is of the form :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = \operatorname{min}(x_i, x_i')\quad\text{with}\quad x_i, x_i' \geq 0,

    :math:`d` is the input dimensionality,
    and :math:`\sigma^2` is the ``variance`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IProductBrownian`.
        Each subclass of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureKernel`

    :param brownian_kernel: The standard EmuKit product Brownian kernel.
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
        brownian_kernel: IProductBrownian,
        integral_bounds: Optional[BoundsType],
        measure: Optional[IntegrationMeasure],
        variable_names: str = "",
    ) -> None:
        super().__init__(
            kern=brownian_kernel, integral_bounds=integral_bounds, measure=measure, variable_names=variable_names
        )

        lower_bounds_x = self.reasonable_box.lower_bounds[0, :]
        if any(lower_bounds_x < 0):
            raise ValueError(
                "The domain defined by the reasonable box seems to allow negative values. "
                "Brownian motion is only defined for positive input values."
            )

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
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


class QuadratureProductBrownianLebesgueMeasure(QuadratureProductBrownian):
    """An product Brownian kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureProductBrownian`

    :param brownian_kernel: The standard EmuKit product Brownian kernel.
    :param integral_bounds: The integral bounds.
                            List of D tuples, where D is the dimensionality
                            of the integral and the tuples contain the lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            ``None`` if bounds are infinite.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self, brownian_kernel: IProductBrownian, integral_bounds: BoundsType, variable_names: str = ""
    ) -> None:
        super().__init__(
            brownian_kernel=brownian_kernel,
            integral_bounds=integral_bounds,
            measure=None,
            variable_names=variable_names,
        )

    def qK(self, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        if skip is None:
            skip = []

        qK = np.ones(x2.shape[0])
        for dim in range(x2.shape[1]):
            if dim in skip:
                continue
            qK *= self._qK_1d(x=x2[:, dim], domain=self.integral_bounds.bounds[dim])
        return qK[None, :] * self.variance

    def qKq(self) -> float:
        qKq = 1.0
        for dim in range(self.input_dim):
            qKq *= self._qKq_1d(domain=self.integral_bounds.bounds[dim])
        return self.variance * qKq

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        input_dim = x2.shape[1]
        dqK_dx = np.zeros([input_dim, x2.shape[0]])
        for dim in range(input_dim):
            grad_term = self._dqK_dx_1d(x=x2[:, dim], domain=self.integral_bounds.bounds[dim])
            dqK_dx[dim, :] = grad_term * self.qK(x2, skip=[dim])[0, :]
        return dqK_dx

    # one dimensional integrals start here
    def _qK_1d(self, x: np.ndarray, domain: Tuple[float, float]) -> np.ndarray:
        """Unscaled kernel mean for 1D Brownian kernel."""
        (a, b) = domain
        kernel_mean = b * x - 0.5 * x**2 - 0.5 * a**2
        return kernel_mean.T

    def _qKq_1d(self, domain: Tuple[float, float]) -> float:
        """Unscaled kernel variance for 1D Brownian kernel."""
        a, b = domain
        qKq = 0.5 * b * (b**2 - a**2) - (b**3 - a**3) / 6 - 0.5 * a**2 * (b - a)
        return float(qKq)

    def _dqK_dx_1d(self, x, domain):
        """Unscaled gradient of 1D Brownian kernel mean."""
        _, b = domain
        return (b - x).T
