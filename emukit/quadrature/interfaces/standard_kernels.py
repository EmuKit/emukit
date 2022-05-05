# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


class IStandardKernel:
    """Interface for a standard kernel k(x, x') that in principle can be integrated.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.interfaces.IProductMatern32`

    """

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :returns: Kernel evaluated at x1, x2, shape (N, M).
        """
        raise NotImplementedError

    # the following methods are gradients of the kernel
    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Gradient of the kernel wrt x1 evaluated at pair x1, x2.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M)
        """
        raise NotImplementedError

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """The gradient of the diagonal of the kernel (the variance) v(x):=k(x, x) evaluated at x.

        :param x: The locations where the gradient is evaluated, shape (n_points, input_dim).
        :return: The gradient of the diagonal of the kernel evaluated at x, shape (input_dim, n_points).
        """
        raise NotImplementedError


class IRBF(IStandardKernel):
    r"""Interface for an RBF kernel.


    .. math::
        k(x, x') = \sigma^2 e^{-\frac{1}{2}\frac{\|x-x'\|^2}{\lambda^2}},

    where :math:`\sigma^2` is the ``variance`` property and :math:`\lambda` is the
    ``lengthscale`` property.

    .. note::
        Inherit from this class to wrap your standard RBF kernel. The wrapped kernel can then be
        handed to a quadrature RBF kernel that augments it with integrability.

    .. seealso::
       * :class:`emukit.quadrature.kernels.QuadratureRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBFLebesgueMeasure`
       * :class:`emukit.quadrature.kernels.QuadratureRBFIsoGaussMeasure`
       * :class:`emukit.quadrature.kernels.QuadratureRBFUniformMeasure`

    """

    @property
    def lengthscale(self) -> float:
        r"""The lengthscale :math:`\lambda` of the kernel."""
        raise NotImplementedError

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        raise NotImplementedError

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        K = self.K(x1, x2)
        scaled_vector_diff = (x1.T[:, :, None] - x2.T[:, None, :]) / self.lengthscale**2
        dK_dx1 = -K[None, ...] * scaled_vector_diff
        return dK_dx1

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[1], x.shape[0]))


class IProductMatern32(IStandardKernel):
    r"""Interface for a Matern32 product kernel.

    Inherit from this class to wrap your ProductMatern32 kernel.

    The product kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{3}r_i ) e^{-\sqrt{3} r_i}.

    :math:`d` is the input dimensionality,
    :math:`r_i:=\frac{|x_i - x'_i|}{\lambda_i}`,
    :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i` is the :math:`i` th element
    of the ``lengthscales`` property.

    Make sure to encode only a single variance parameter, and not one for each individual :math:`k_i`.

    .. note::
        Inherit from this class to wrap your standard product Matern32 kernel. The wrapped kernel can then be
        handed to a quadrature product Matern32 kernel that augments it with integrability.

    .. seealso::
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern32`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern32LebesgueMeasure`

    """

    @property
    def nu(self) -> float:
        """The smoothness parameter of the kernel."""
        return 1.5

    @property
    def lengthscales(self) -> np.ndarray:
        r"""The lengthscales :math:`\lambda` of the kernel."""
        raise NotImplementedError

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        raise NotImplementedError

    def _dK_dx1_1d(self, x1: np.ndarray, x2: np.ndarray, ell: float) -> np.ndarray:
        """Unscaled gradient of 1D Matern32 where ``ell`` is the lengthscale parameter.

        This method can be used in case the product Matern32 is implemented via a List of
        univariate Matern32 kernels.

        :param x1: First argument of the kernel, shape = (n_points N,).
        :param x2: Second argument of the kernel, shape = (n_points M,).
        :param ell: The lengthscale of the 1D Matern32.
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (N, M).
        """
        r = (x1.T[:, None] - x2.T[None, :]) / ell  # N x M
        dr_dx1 = r / (ell * abs(r))
        dK_dr = -3 * abs(r) * np.exp(-np.sqrt(3) * abs(r))
        return dK_dr * dr_dx1

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[1], x.shape[0]))


class IProductMatern52(IStandardKernel):
    r"""Interface for a Matern52 product kernel.

    Inherit from this class to wrap your ProductMatern52 kernel.

    The product kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{5} r_i + \frac{5}{3} r_i^2) \exp(- \sqrt{5} r_i).

    :math:`d` is the input dimensionality,
    :math:`r_i:=\frac{|x_i - x'_i|}{\lambda_i}`,
    :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i` is the :math:`i` th element
    of the ``lengthscales`` property.

    Make sure to encode only a single variance parameter, and not one for each individual :math:`k_i`.

    .. note::
        Inherit from this class to wrap your standard product Matern52 kernel. The wrapped kernel can then be
        handed to a quadrature product Matern52 kernel that augments it with integrability.

    .. seealso::
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern52`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern52LebesgueMeasure`

    """

    @property
    def nu(self) -> float:
        """The smoothness parameter of the kernel."""
        return 2.5

    @property
    def lengthscales(self) -> np.ndarray:
        r"""The lengthscales :math:`\lambda` of the kernel."""
        raise NotImplementedError

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        raise NotImplementedError

    def _dK_dx1_1d(self, x1: np.ndarray, x2: np.ndarray, ell: float) -> np.ndarray:
        """Unscaled gradient of 1D Matern52 where ``ell`` is the lengthscale parameter.

        This method can be used in case the product Matern52 is implemented via a List of
        univariate Matern52 kernels.

        :param x1: First argument of the kernel, shape = (n_points N,).
        :param x2: Second argument of the kernel, shape = (n_points M,).
        :param ell: The lengthscale of the 1D Matern52.
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (N, M).
        """
        r = (x1.T[:, None] - x2.T[None, :]) / ell  # N x M
        dr_dx1 = r / (ell * abs(r))
        dK_dr = (-5 / 3) * np.exp(-np.sqrt(5) * abs(r)) * (abs(r) + np.sqrt(5) * r**2)
        return dK_dr * dr_dx1

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[1], x.shape[0]))


class IBrownian(IStandardKernel):
    r"""Interface for a Brownian motion kernel.

    .. math::
        k(x, x') = \sigma^2 \operatorname{min}(x, x')\quad\text{with}\quad x, x' \geq 0,

    where :math:`\sigma^2` is the ``variance`` property.

    .. note::
        Inherit from this class to wrap your standard Brownian motion kernel.
        The wrapped kernel can then be handed to a quadrature Brownian motion kernel that
        augments it with integrability.

    .. seealso::
       * :class:`emukit.quadrature.kernels.QuadratureBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureBrownianLebesgueMeasure`

    """

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        raise NotImplementedError

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1_rep = np.repeat(x1[:, 0][np.newaxis, ...], x2.shape[0], axis=0).T
        x2_rep = np.repeat(x2[:, 0][np.newaxis, ...], x1.shape[0], axis=0)
        return self.variance * (x1_rep < x2_rep)[np.newaxis, :, :]

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        return self.variance * np.ones((x.shape[1], x.shape[0]))


class IProductBrownian(IStandardKernel):
    r"""Interface for a Brownian product kernel.

    Inherit from this class to wrap your ProductBrownian kernel.

    The product kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = \operatorname{min}(x_i, x_i')\quad\text{with}\quad x_i, x_i' \geq 0,

    :math:`d` is the input dimensionality,
    and :math:`\sigma^2` is the ``variance`` property.

    Make sure to encode only a single variance parameter, and not one for each individual :math:`k_i`.

    .. note::
        Inherit from this class to wrap your standard product Brownian kernel. The wrapped kernel can then be
        handed to a quadrature product Brownian kernel that augments it with integrability.

    .. seealso::
       * :class:`emukit.quadrature.kernels.QuadratureProductBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureProductBrownianLebesgueMeasure`

    """

    @property
    def variance(self) -> float:
        r"""The scale :math:`\sigma^2` of the kernel."""
        raise NotImplementedError

    def _dK_dx1_1d(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Unscaled gradient of 1D Brownian kernel.

        This method can be used in case the product Brownian is implemented via a List of
        Brownian kernels.

        :param x1: First argument of the kernel, shape = (n_points N,).
        :param x2: Second argument of the kernel, shape = (n_points M,).
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (N, M).
        """
        x1_rep = np.repeat(x1[:, 0][np.newaxis, ...], x2.shape[0], axis=0).T
        x2_rep = np.repeat(x2[:, 0][np.newaxis, ...], x1.shape[0], axis=0)
        return (x1_rep < x2_rep)[np.newaxis, :, :]

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        return self.variance * np.ones((x.shape[1], x.shape[0]))
