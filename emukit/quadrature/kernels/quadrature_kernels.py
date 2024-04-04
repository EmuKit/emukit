# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Base class for kernel embeddings."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union

import numpy as np

from ..interfaces.standard_kernels import IStandardKernel
from ..measures import BoxDomain, GaussianMeasure, IntegrationMeasure, LebesgueMeasure
from ..typing import BoundsType


class QuadratureKernel:
    """Abstract class for a kernel augmented with integrability.

    .. note::

        Each specific implementation of this class must go with a specific standard kernel as input which
        inherits from :class:`IStandardKernel`. This is because we both want the :class:`QuadratureKernel`
        to be backend agnostic but at the same time :class:`QuadratureKernel` needs access to specifics
        of the standard kernel. For example a specific pair of :class:`QuadratureKernel` and
        :class:`IStandardKernel` is :class:`QuadratureRBF` and :class:`IRBF`. The kernel embeddings are
        implemented w.r.t. a specific integration measure, for example the :class:`LebesgueMeasure`.

    :param kern: Standard EmuKit kernel.
    :param measure: The integration measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        kern: IStandardKernel,
        measure: IntegrationMeasure,
        variable_names: str,
    ) -> None:
        self.kern = kern
        self.measure = measure
        self.input_dim = self.measure.input_dim

        # Even though the domain of the integration might be unbounded, some subroutines such as the acquisition
        # optimizer require a bounded box. Hence, below we define a reasonable box for input values of the integrand
        # that can be used by such methods. The box is defined by the integration measure used (see docs there).
        # Note that the reasonable box only affects methods that use this box, e.g. the acqusition optimizers.
        # The kernel embeddings still use the originally defined domain which may be unbounded.
        self.reasonable_box = BoxDomain(name=variable_names, bounds=measure.reasonable_box())

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2.

        :param x1: First argument of the kernel.
        :param x2: Second argument of the kernel.
        :returns: The kernel at x1, x2.
        """
        return self.kern.K(x1, x2)

    # the following methods are integrals of a quadrature kernel
    def qK(self, x2: np.ndarray) -> np.ndarray:
        """The kernel with the first argument integrated out (kernel mean) evaluated at x2.

        :param x2: The locations where the kernel mean is evaluated, shape (n_points, input_dim).
        :returns: The kernel mean at x2, shape (1, n_points).
        """
        raise NotImplementedError

    def Kq(self, x1: np.ndarray) -> np.ndarray:
        """The kernel with the second argument integrated out (kernel mean) evaluated at x1.

        :param x1: The locations where the kernel mean is evaluated, shape (n_points, input_dim).
        :returns: The kernel mean at x1, shape (n_points, 1).
        """
        return self.qK(x1).T

    def qKq(self) -> float:
        """The kernel integrated over both arguments x1 and x2.

        :returns: Double integrated kernel.
        """
        raise NotImplementedError

    # the following methods are gradients of a quadrature kernel
    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The gradient of the kernel wrt x1 evaluated at pair x1, x2.

        :param x1: First argument of the kernel, shape (n_points N, input_dim).
        :param x2: Second argument of the kernel, shape (n_points M, input_dim).
        :return: The gradient at (x1, x2), shape (input_dim, N, M).
        """
        return self.kern.dK_dx1(x1, x2)

    def dK_dx2(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The gradient of the kernel wrt x2 evaluated at pair x1, x2.

        Note that this is equal to the transposed gradient of the kernel wrt x1 evaluated
        at x2 and x1 (swapped arguments).

        :param x1: First argument of the kernel, shape (n_points N, N, input_dim).
        :param x2: Second argument of the kernel, shape (n_points N, M, input_dim).
        :return: The gradient at (x1, x2), shape (input_dim, N, M).
        """
        return np.transpose(self.dK_dx1(x1=x2, x2=x1), (0, 2, 1))

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """The gradient of the diagonal of the kernel :math:`v(x):=k(x, x)` evaluated at x.

        :param x: The locations where the gradient is evaluated, shape = (n_points M, input_dim).
        :return: The gradient at x, shape (input_dim, M).
        """
        return self.kern.dKdiag_dx(x)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """The gradient of the kernel mean (integrated in first argument) evaluated at x2.

        :param x2: The locations where the gradient is evaluated, shape (n_points N, N, input_dim).
        :return: The gradient at x2, shape (input_dim, N).
        """
        raise NotImplementedError

    def dKq_dx(self, x1: np.ndarray) -> np.ndarray:
        """The gradient of the kernel mean (integrated in second argument) evaluated at x1.

        :param x1: The locations where the gradient is evaluated, shape (n_points N, N, input_dim).
        :return: The gradient with shape (N, input_dim).
        """
        return self.dqK_dx(x1).T


class QuadratureProductKernel(QuadratureKernel):
    """Abstract class for a product kernel augmented with integrability.

    The product kernel is of the form :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')`
    where :math:`k_i(x, x')` is a univariate kernel acting on dimension :math:`i`.

    :param kern: Standard EmuKit kernel (must be a product kernel).
    :param measure: The integration measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(
        self,
        kern: IStandardKernel,
        measure: IntegrationMeasure,
        variable_names: str,
    ) -> None:
        super().__init__(kern=kern, measure=measure, variable_names=variable_names)

    def _qK_unscaled(self, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        if skip is None:
            skip = []

        qK = np.ones(x2.shape[0])
        for dim in range(x2.shape[1]):
            if dim in skip:
                continue
            qK *= self._qK_1d(x2[:, dim], **self._get_univariate_parameters(dim))
        return qK[None, :]

    def qK(self, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        return self._scale(self._qK_unscaled(x2, skip))

    def qKq(self) -> float:
        qKq = 1.0
        for dim in range(self.input_dim):
            qKq *= self._qKq_1d(**self._get_univariate_parameters(dim))
        return self._scale(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        input_dim = x2.shape[1]
        dqK_dx = np.zeros([input_dim, x2.shape[0]])
        for dim in range(input_dim):
            grad_term = self._dqK_dx_1d(x2[:, dim], **self._get_univariate_parameters(dim))
            dqK_dx[dim, :] = grad_term * self._qK_unscaled(x2, skip=[dim])[0, :]
        return self._scale(dqK_dx)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Scales the input ``z`` with a scalar value specific to the kernel.

        :param z: The value to be scaled.
        :return: The scaled value.
        """
        raise NotImplementedError

    def _get_univariate_parameters(self, dim: int) -> dict:
        """Keywords arguments used by methods related to the univariate, unscaled version
         of the kernel of dimension ``dim``.

        :param dim: The dimension.
        :return: The parameters of dimension ``dim``.
        """
        raise NotImplementedError

    # methods related to the univariate version of the kernel
    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        """Unscaled kernel mean for univariate version of kernel.

        :param x: The locations where the kernel mean is evaluated, shape (n_points, ).
        :param parameters: The parameters of the univariate kernel.
        :return: The kernel mean of the univariate kernel evaluated at x, shape (n_points, ).
        """
        raise NotImplementedError

    def _qKq_1d(self, **parameters) -> float:
        """Unscaled kernel integrated over both arguments for univariate version of kernel.

        :param parameters: The parameters of the univariate kernel.
        :returns: Double integrated univariate kernel.
        """
        raise NotImplementedError

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        """Unscaled gradient of univariate version of the kernel mean.

        :param x: The locations where the kernel mean is evaluated, shape (n_points, ).
        :param parameters: The parameters of the univariate kernel.
        :return: The gradient with shape (n_points, ).
        """
        raise NotImplementedError


class LebesgueEmbedding:
    """Mixin for quadrature kernel w.r.t. Lebesgue measure.

    .. seealso::
        * :class:`emukit.quadrature.measures.LebesgueMeasure`

    """

    @classmethod
    def from_integral_bounds(
        cls, kern: IStandardKernel, integral_bounds: BoundsType, normalized: bool = False, variable_names: str = ""
    ):
        """Create the quadrature kernel w.r.t. a Lebesgue measure from the integral bounds.

        :param kern: Standard EmuKit kernel.
        :param integral_bounds: The integral bounds.
                               List of :math:`d` tuples :math:`[(a_1, b_1), (a_2, b_2), \dots, (a_d, b_d)]`,
                               where :math:`d` is the dimensionality of the integral and the tuple :math:`(a_i, b_i)`
                               contains the lower and upper integration bound of dimension :math:`i`.
        :param normalized: Weather the Lebesgue measure is normalized.
        :param variable_names: The (variable) name(s) of the integral
        :return: An instance of a quadrature kernel w.r.t. Lebesgue measure.

        """
        domain = BoxDomain(bounds=integral_bounds)
        measure = LebesgueMeasure(domain=domain, normalized=normalized)
        return cls(kern, measure, variable_names)


class GaussianEmbedding:
    """Mixin for quadrature kernel w.r.t. Gaussian measure.

    .. seealso::
        * :class:`emukit.quadrature.measures.GaussianMeasure`

    """

    @classmethod
    def from_measure_params(
        cls, kern: IStandardKernel, mean: np.ndarray, variance: Union[float, np.ndarray], variable_names: str = ""
    ):
        """Create the quadrature kernel w.r.t. a Gaussian measure from the measure parameters.

        :param kern: Standard EmuKit kernel.
        :param mean: The mean of the Gaussian measure, shape (input_dim, ).
        :param variance: The variances of the Gaussian measure, shape (input_dim, ).
                         If a scalar value is given, all dimensions will have same variance.
        :param variable_names: The (variable) name(s) of the integral
        :return: An instance of a quadrature kernel w.r.t. Gaussian measure.

        """
        measure = GaussianMeasure(mean=mean, variance=variance)
        return cls(kern, measure, variable_names)
