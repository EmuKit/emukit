"""The Brownian motion kernel embeddings."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Union

from ...quadrature.interfaces.standard_kernels import IBrownian, IProductBrownian
from ..kernels import QuadratureKernel, QuadratureProductKernel, LebesgueEmbedding
from ..measures import IntegrationMeasure, LebesgueMeasure


class QuadratureBrownian(QuadratureKernel):
    r"""Base class for a Brownian motion kernel augmented with integrability.

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
    :param measure: The integration measure.
    :param variable_names: The (variable) name(s) of the integral.

    :raises ValueError: If ``integral_bounds`` have wrong length.

    """

    def __init__(
        self,
        brownian_kernel: IBrownian,
        measure: IntegrationMeasure,
        variable_names: str,
    ) -> None:

        if measure.input_dim != 1:
            raise ValueError("Integration measure for Brownian motion kernel must be 1-dimensional. Current dimesnion is ({}).".format(measure.input_dim))

        super().__init__(kern=brownian_kernel, measure=measure, variable_names=variable_names)

        if any(self.reasonable_box.lower_bounds < 0):
            raise ValueError(
                "The domain defined by the reasonable box seems to allow negative values. "
                "Brownian motion is only defined for positive input values."
            )

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        return self.kern.variance


class QuadratureBrownianLebesgueMeasure(QuadratureBrownian, LebesgueEmbedding):
    """A Brownian motion kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureBrownian`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param brownian_kernel: The standard EmuKit Brownian motion kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, brownian_kernel: IBrownian, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(brownian_kernel=brownian_kernel, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        kernel_mean = ub * x2 - 0.5 * x2**2 - 0.5 * lb**2
        return self.variance * kernel_mean.T

    def qKq(self) -> float:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        qKq = 0.5 * ub * (ub**2 - lb**2) - (ub**3 - lb**3) / 6 - 0.5 * lb**2 * (ub - lb)
        return float(self.variance * qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        ub = self.measure.domain.upper_bounds[None, :]
        return self.variance * (ub - x2).T


class QuadratureProductBrownian(QuadratureProductKernel):
    r"""Base class for a product Brownian kernel augmented with integrability.

    The kernel is of the form :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = \operatorname{min}(x_i-c, x_i'-c)\quad\text{with}\quad x_i, x_i' \geq c,

    :math:`d` is the input dimensionality,
    :math:`\sigma^2` is the ``variance`` property
    and :math:`c` is the ``offset`` property.

    .. note::
        This class is compatible with the standard kernel :class:`IProductBrownian`.
        Each subclass of this class implements an embedding w.r.t. a specific integration measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureProductKernel`

    :param brownian_kernel: The standard EmuKit product Brownian kernel.
    :param measure: The integration measure.
    :param variable_names: The (variable) name(s) of the integral.
    """

    def __init__(
        self,
        brownian_kernel: IProductBrownian,
        measure: IntegrationMeasure,
        variable_names: str,
    ) -> None:
        super().__init__(
            kern=brownian_kernel, measure=measure, variable_names=variable_names
        )

        lower_bounds_x = self.reasonable_box.lower_bounds[0, :]
        if any(lower_bounds_x < self.offset):
            raise ValueError(
                f"The domain defined by the reasonable box seems allow to for values smaller than the offset "
                f"({self.offset}). Brownian motion is only defined for input values larger than the offset."
            )

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        return self.kern.variance

    @property
    def offset(self) -> float:
        r"""The offset :math:`c` of the kernel."""
        return self.kern.offset


class QuadratureProductBrownianLebesgueMeasure(QuadratureProductBrownian, LebesgueEmbedding):
    """A product Brownian kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureProductBrownian`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param brownian_kernel: The standard EmuKit product Brownian kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, brownian_kernel: IProductBrownian, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(brownian_kernel=brownian_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        return {"domain": self.measure.domain.bounds[dim], "offset": self.offset}

    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        offset = parameters["offset"]
        kernel_mean = b * x - 0.5 * x**2 - 0.5 * a**2
        return kernel_mean.T - offset * (b - a)

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        offset = parameters["offset"]
        qKq = 0.5 * b * (b**2 - a**2) - (b**3 - a**3) / 6 - 0.5 * a**2 * (b - a)
        return float(qKq) - offset * (b - a) ** 2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        _, b = parameters["domain"]
        return (b - x).T
