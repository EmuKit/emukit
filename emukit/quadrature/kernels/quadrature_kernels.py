"""Base class for kernel embeddings."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Tuple

import numpy as np

from emukit.quadrature.interfaces.standard_kernels import IStandardKernel
from emukit.quadrature.measures import BoxDomain, IntegrationMeasure


class QuadratureKernel:
    """Abstract class for a kernel augmented with integrability.

    Note that each specific implementation of this class must go with a specific standard kernel as input which
    inherits from :class:`IStandardKernel`. This is because we both want the :class:`QuadratureKernel` to be backend
    agnostic but at the same time :class:`QuadratureKernel` needs access to specifics of the standard kernel.
    For example a specific pair of :class:`QuadratureKernel` and :class:`IStandardKernel` is :class:`QuadratureRBF`
    and :class:`IRBF`. The kernel embeddings are implemented w.r.t. a specific integration measure, for example
    the :class:`LebesgueMeasure`.

    """

    def __init__(
        self,
        kern: IStandardKernel,
        integral_bounds: Optional[List[Tuple[float, float]]],
        measure: Optional[IntegrationMeasure],
        variable_names: str = "",
    ) -> None:
        """
        :param kern: Standard emukit kernel.
        :param integral_bounds: The integral bounds.
                                List of D tuples, where D is the dimensionality
                                of the integral and the tuples contain the lower and upper bounds of the integral
                                i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                                ``None`` if bounds are infinite.
        :param measure: The integration measure. ``None`` implies the standard Lebesgue measure.
        :param variable_names: The (variable) name(s) of the integral
        """

        # we define reasonable box bounds for the integral because e.g., the optimizer of the acquisition function
        # requires finite bounds. The box is defined by the integration measure. See each integration measure for
        # details. Note that this only affects methods that use this box, e.g. the acqusition optimizers. The integrals
        # in this kernel will have infinite bounds still.
        if (integral_bounds is None) and (measure is None):
            raise ValueError("integral_bounds and measure are both None. At least one of them must be given.")
        if integral_bounds is None:
            reasonable_box_bounds = measure.get_box()
            self.integral_bounds = None
        else:
            reasonable_box_bounds = integral_bounds
            self.integral_bounds = BoxDomain(name=variable_names, bounds=integral_bounds)

        self.reasonable_box = BoxDomain(name=variable_names, bounds=reasonable_box_bounds)
        self.kern = kern
        self.measure = measure
        self.input_dim = self.reasonable_box.dim

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
        raise NotImplementedError

    def qKq(self) -> np.float:
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
        raise NotImplementedError
