# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""GPy wrappers for the quadrature package."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import List, Optional, Tuple, Union

import GPy
import numpy as np
from scipy.linalg import lapack

from ..quadrature.interfaces import (
    IRBF,
    IBaseGaussianProcess,
    IBrownian,
    IProductBrownian,
    IProductMatern12,
    IProductMatern32,
    IProductMatern52,
)
from ..quadrature.kernels import (
    QuadratureBrownianLebesgueMeasure,
    QuadratureKernel,
    QuadratureProductBrownian,
    QuadratureProductBrownianLebesgueMeasure,
    QuadratureProductMatern12LebesgueMeasure,
    QuadratureProductMatern32LebesgueMeasure,
    QuadratureProductMatern52LebesgueMeasure,
    QuadratureRBFGaussianMeasure,
    QuadratureRBFLebesgueMeasure,
)
from ..quadrature.measures import BoxDomain, GaussianMeasure, IntegrationMeasure, LebesgueMeasure
from ..quadrature.typing import BoundsType


class BaseGaussianProcessGPy(IBaseGaussianProcess):
    """Wrapper for GPy's :class:`GPRegression` as required for some EmuKit quadrature methods.

    An instance of this class can be passed as :attr:`base_gp` to a :class:`WarpedBayesianQuadratureModel` object.

    .. note::
        GPy's :class:`GPRegression` cannot take ``None`` as initial values for X and Y. Thus, we initialize
        them with some arbitrary values. These will be re-set in the :class:`WarpedBayesianQuadratureModel`.

    :param kern: An EmuKit quadrature kernel.
    :param gpy_model: A GPy GP regression model.
    :param noise_free: If ``False``, the observation noise variance will be treated as a model parameter,
                       if ``True`` the noise is set to 1e-10, defaults to ``True``.
    """

    def __init__(self, kern: QuadratureKernel, gpy_model: GPy.models.GPRegression, noise_free: bool = True):
        super().__init__(kern=kern)
        if noise_free:
            gpy_model.Gaussian_noise.constrain_fixed(1.0e-10)
        self.gpy_model = gpy_model

        if isinstance(kern, QuadratureProductBrownian):
            if kern.offset != 0:
                raise ValueError(
                    "The wrapper BaseGaussianProcessGPy does not support EmuKit product Brownian "
                    "motion kernels with non-zero offset as these are not supported in GPy."
                )

    @property
    def X(self) -> np.ndarray:
        """The data nodes."""
        return self.gpy_model.X

    @property
    def Y(self) -> np.ndarray:
        """The data evaluations at the nodes."""
        return self.gpy_model.Y

    @property
    def observation_noise_variance(self) -> float:
        return self.gpy_model.Gaussian_noise[0]

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Sets training data in model.

        :param X: New training features, shape (num_points, input_dim).
        :param Y: New training outputs, shape (num_points, 1).
        """
        self.gpy_model.set_XY(X, Y)

    def predict(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean and covariance at the locations X_pred.

        :param X_pred: Points at which to predict, with shape (number of points, input_dim).
        :return: Predictive mean, predictive variances shapes (num_points, 1) and (num_points, 1).
        """
        return self.gpy_model.predict(X_pred, full_cov=False)

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean and covariance at the locations X_pred.

        :param X_pred: Points at which to predict, with shape (num_points, input_dim).
        :return: Predictive mean, predictive full covariance shapes (num_points, 1) and (num_points, num_points).
        """
        return self.gpy_model.predict(X_pred, full_cov=True)

    def solve_linear(self, z: np.ndarray) -> np.ndarray:
        lower_chol = self.gpy_model.posterior.woodbury_chol
        return lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, z, lower=1)[0]), lower=0)[0]

    def graminv_residual(self) -> np.ndarray:
        return self.gpy_model.posterior.woodbury_vector

    def optimize(self) -> None:
        """Optimize the hyperparameters of the GP."""
        self.gpy_model.optimize()


class RBFGPy(IRBF):
    r"""Wrapper of the GPy RBF kernel as required for some EmuKit quadrature methods.

    .. math::
        k(x, x') = \sigma^2 e^{-\frac{1}{2}\sum_{i=1}^{d}r_i^2},

    where :math:`d` is the input dimensionality,
    :math:`r_i = \frac{x_i-x_i'}{\lambda_i}` is the scaled vector difference of dimension :math:`i`,
    :math:`\lambda_i` is the :math:`i` th element of the ``lengthscales`` property
    and :math:`\sigma^2` is the ``variance`` property.

    :param gpy_rbf: An RBF kernel from GPy.
    """

    def __init__(self, gpy_rbf: GPy.kern.RBF):
        self.gpy_rbf = gpy_rbf

    @property
    def lengthscales(self) -> np.ndarray:
        if self.gpy_rbf.ARD:
            return self.gpy_rbf.lengthscale.values
        return np.full((self.gpy_rbf.input_dim,), self.gpy_rbf.lengthscale[0])

    @property
    def variance(self) -> float:
        return self.gpy_rbf.variance.values[0]

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_rbf.K(x1, x2)


class ProductMatern12GPy(IProductMatern12):
    r"""Wrapper of the GPy Exponential (a.k.a Matern12) product kernel as required for some
    EmuKit quadrature methods.

    The product kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = e^{-r_i}.

    :math:`d` is the input dimensionality,
    :math:`r_i:=\frac{|x_i - x'_i|}{\lambda_i}`,
    :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i` is the :math:`i` th element
    of the ``lengthscales`` property.

    :param gpy_matern: An Exponential (a.k.a. Matern12) product kernel from GPy. For :math:`d=1` this is equivalent
                       to an Exponential kernel. For :math:`d>1`, this is *not* a :math:`d`-dimensional
                       Exponential kernel but a product of :math:`d` 1-dimensional Exponential kernels with differing
                       active dimensions constructed as k1 * k2 * ... .
                       Make sure to unlink all variances except the variance of the first kernel k1 in the product
                       as the variance of k1 will be used to represent :math:`\sigma^2`. If you are unsure what
                       to do, use the :attr:`lengthscales` and :attr:`variance` parameter instead.
                       If :attr:`gpy_matern` is not given, the :attr:`lengthscales` argument is used.
    :param lengthscales: If :attr:`gpy_matern` is not given, a product Matern12 kernel will be constructed with
                       the given lengthscales. The number of elements need to be equal to the dimensionality
                       :math:`d`. If :attr:`gpy_matern` is given, this input is disregarded.
    :param variance: The variance of the product kernel. Only used if :attr:`gpy_matern` is not given. Defaults to 1.
    """

    def __init__(
        self,
        gpy_matern: Optional[Union[GPy.kern.Exponential, GPy.kern.Prod]] = None,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[float] = None,
    ):
        if gpy_matern is None and lengthscales is None:
            raise ValueError("Either lengthscales or a GPy product matern kernel must be given.")

        # product kernel from parameters
        if gpy_matern is None:
            input_dim = len(lengthscales)
            if input_dim < 1:
                raise ValueError("'lengthscales' must contain at least 1 value.")

            # default variance
            if variance is None:
                variance = 1.0

            gpy_matern = GPy.kern.Exponential(
                input_dim=1, active_dims=[0], lengthscale=lengthscales[0], variance=variance
            )
            for dim in range(1, input_dim):
                k = GPy.kern.Exponential(input_dim=1, active_dims=[dim], lengthscale=lengthscales[dim])
                k.unlink_parameter(k.variance)
                gpy_matern = gpy_matern * k

        self.gpy_matern = gpy_matern

    @property
    def lengthscales(self) -> np.ndarray:
        if isinstance(self.gpy_matern, GPy.kern.Exponential):
            return np.array([self.gpy_matern.lengthscale[0]])

        lengthscales = []
        for kern in self.gpy_matern.parameters:
            lengthscales.append(kern.lengthscale[0])
        return np.array(lengthscales)

    @property
    def variance(self) -> float:
        if isinstance(self.gpy_matern, GPy.kern.Exponential):
            return self.gpy_matern.variance[0]

        return self.gpy_matern.parameters[0].variance[0]

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_matern.K(x1, x2)

    def _K_from_prod(self, x1: np.ndarray, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2 computed as product from the
        individual 1d kernels.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :param skip: Skip these dimensions if specified.
        :returns: Kernel evaluated at x1, x2, shape (N, M).
        """
        if skip is None:
            skip = []
        K = np.ones([x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_matern.parameters):
            if dim in skip:
                continue
            K *= kern.K(x1, x2)

        # correct for missing variance
        if 0 in skip:
            K *= self.variance
        return K

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if isinstance(self.gpy_matern, GPy.kern.Exponential):
            return self._dK_dx1_1d(x1[:, 0], x2[:, 0], self.gpy_matern.lengthscale[0])[None, :, :]

        # product kernel
        dK_dx1 = np.ones([x1.shape[1], x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_matern.parameters):
            prod_term = self._K_from_prod(x1, x2, skip=[dim])  # N x M
            grad_term = self._dK_dx1_1d(x1[:, dim], x2[:, dim], kern.lengthscale[0])  # N x M
            dK_dx1[dim, :, :] *= prod_term * grad_term
        return dK_dx1


class ProductMatern32GPy(IProductMatern32):
    r"""Wrapper of the GPy Matern32 product kernel as required for some EmuKit quadrature methods.

    The product kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{3}r_i ) e^{-\sqrt{3} r_i}.

    :math:`d` is the input dimensionality,
    :math:`r_i:=\frac{|x_i - x'_i|}{\lambda_i}`,
    :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i` is the :math:`i` th element
    of the ``lengthscales`` property.

    :param gpy_matern: A Matern32 product kernel from GPy. For :math:`d=1` this is equivalent to a
                       Matern32 kernel. For :math:`d>1`, this is *not* a :math:`d`-dimensional
                       Matern32 kernel but a product of :math:`d` 1-dimensional Matern32 kernels with differing
                       active dimensions constructed as k1 * k2 * ... .
                       Make sure to unlink all variances except the variance of the first kernel k1 in the product
                       as the variance of k1 will be used to represent :math:`\sigma^2`. If you are unsure what
                       to do, use the :attr:`lengthscales` and :attr:`variance` parameter instead.
                       If :attr:`gpy_matern` is not given, the :attr:`lengthscales` argument is used.
    :param lengthscales: If :attr:`gpy_matern` is not given, a product Matern32 kernel will be constructed with
                       the given lengthscales. The number of elements need to be equal to the dimensionality
                       :math:`d`. If :attr:`gpy_matern` is given, this input is disregarded.
    :param variance: The variance of the product kernel. Only used if :attr:`gpy_matern` is not given. Defaults to 1.
    """

    def __init__(
        self,
        gpy_matern: Optional[Union[GPy.kern.Matern32, GPy.kern.Prod]] = None,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[float] = None,
    ):
        if gpy_matern is None and lengthscales is None:
            raise ValueError("Either lengthscales or a GPy product matern kernel must be given.")

        # product kernel from parameters
        if gpy_matern is None:
            input_dim = len(lengthscales)
            if input_dim < 1:
                raise ValueError("'lengthscales' must contain at least 1 value.")

            # default variance
            if variance is None:
                variance = 1.0

            gpy_matern = GPy.kern.Matern32(input_dim=1, active_dims=[0], lengthscale=lengthscales[0], variance=variance)
            for dim in range(1, input_dim):
                k = GPy.kern.Matern32(input_dim=1, active_dims=[dim], lengthscale=lengthscales[dim])
                k.unlink_parameter(k.variance)
                gpy_matern = gpy_matern * k

        self.gpy_matern = gpy_matern

    @property
    def lengthscales(self) -> np.ndarray:
        if isinstance(self.gpy_matern, GPy.kern.Matern32):
            return np.array([self.gpy_matern.lengthscale[0]])

        lengthscales = []
        for kern in self.gpy_matern.parameters:
            lengthscales.append(kern.lengthscale[0])
        return np.array(lengthscales)

    @property
    def variance(self) -> float:
        if isinstance(self.gpy_matern, GPy.kern.Matern32):
            return self.gpy_matern.variance[0]

        return self.gpy_matern.parameters[0].variance[0]

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_matern.K(x1, x2)

    def _K_from_prod(self, x1: np.ndarray, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2 computed as product from the
        individual 1d kernels.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :param skip: Skip these dimensions if specified.
        :returns: Kernel evaluated at x1, x2, shape (N, M).
        """
        if skip is None:
            skip = []
        K = np.ones([x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_matern.parameters):
            if dim in skip:
                continue
            K *= kern.K(x1, x2)

        # correct for missing variance
        if 0 in skip:
            K *= self.variance
        return K

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if isinstance(self.gpy_matern, GPy.kern.Matern32):
            return self._dK_dx1_1d(x1[:, 0], x2[:, 0], self.gpy_matern.lengthscale[0])[None, :, :]

        # product kernel
        dK_dx1 = np.ones([x1.shape[1], x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_matern.parameters):
            prod_term = self._K_from_prod(x1, x2, skip=[dim])  # N x M
            grad_term = self._dK_dx1_1d(x1[:, dim], x2[:, dim], kern.lengthscale[0])  # N x M
            dK_dx1[dim, :, :] *= prod_term * grad_term
        return dK_dx1


class ProductMatern52GPy(IProductMatern52):
    r"""Wrapper of the GPy Matern52 product kernel as required for some EmuKit quadrature methods.

    The product kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{5} r_i + \frac{5}{3} r_i^2) \exp(- \sqrt{5} r_i).

    :math:`d` is the input dimensionality,
    :math:`r_i:=\frac{|x_i - x'_i|}{\lambda_i}`,
    :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i` is the :math:`i` th element
    of the ``lengthscales`` property.

    :param gpy_matern: A Matern52 product kernel from GPy. For :math:`d=1` this is equivalent to a
                       Matern52 kernel. For :math:`d>1`, this is *not* a :math:`d`-dimensional
                       Matern52 kernel but a product of :math:`d` 1-dimensional Matern52 kernels with differing
                       active dimensions constructed as k1 * k2 * ... .
                       Make sure to unlink all variances except the variance of the first kernel k1 in the product
                       as the variance of k1 will be used to represent :math:`\sigma^2`. If you are unsure what
                       to do, use the :attr:`lengthscales` and :attr:`variance` parameter instead.
                       If :attr:`gpy_matern` is not given, the :attr:`lengthscales` argument is used.
    :param lengthscales: If :attr:`gpy_matern` is not given, a product Matern52 kernel will be constructed with
                       the given lengthscales. The number of elements need to be equal to the dimensionality
                       :math:`d`. If :attr:`gpy_matern` is given, this input is disregarded.
    :param variance: The variance of the product kernel. Only used if :attr:`gpy_matern` is not given. Defaults to 1.
    """

    def __init__(
        self,
        gpy_matern: Optional[Union[GPy.kern.Matern52, GPy.kern.Prod]] = None,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[float] = None,
    ):
        if gpy_matern is None and lengthscales is None:
            raise ValueError("Either lengthscales or a GPy product matern kernel must be given.")

        # product kernel from parameters
        if gpy_matern is None:
            input_dim = len(lengthscales)
            if input_dim < 1:
                raise ValueError("'lengthscales' must contain at least 1 value.")

            # default variance
            if variance is None:
                variance = 1.0

            gpy_matern = GPy.kern.Matern52(input_dim=1, active_dims=[0], lengthscale=lengthscales[0], variance=variance)
            for dim in range(1, input_dim):
                k = GPy.kern.Matern52(input_dim=1, active_dims=[dim], lengthscale=lengthscales[dim])
                k.unlink_parameter(k.variance)
                gpy_matern = gpy_matern * k

        self.gpy_matern = gpy_matern

    @property
    def lengthscales(self) -> np.ndarray:
        if isinstance(self.gpy_matern, GPy.kern.Matern52):
            return np.array([self.gpy_matern.lengthscale[0]])

        lengthscales = []
        for kern in self.gpy_matern.parameters:
            lengthscales.append(kern.lengthscale[0])
        return np.array(lengthscales)

    @property
    def variance(self) -> float:
        if isinstance(self.gpy_matern, GPy.kern.Matern52):
            return self.gpy_matern.variance[0]

        return self.gpy_matern.parameters[0].variance[0]

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_matern.K(x1, x2)

    def _K_from_prod(self, x1: np.ndarray, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2 computed as product from the
        individual 1d kernels.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :param skip: Skip these dimensions if specified.
        :returns: Kernel evaluated at x1, x2, shape (N, M).
        """
        if skip is None:
            skip = []
        K = np.ones([x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_matern.parameters):
            if dim in skip:
                continue
            K *= kern.K(x1, x2)

        # correct for missing variance
        if 0 in skip:
            K *= self.variance
        return K

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if isinstance(self.gpy_matern, GPy.kern.Matern52):
            return self._dK_dx1_1d(x1[:, 0], x2[:, 0], self.gpy_matern.lengthscale[0])[None, :, :]

        # product kernel
        dK_dx1 = np.ones([x1.shape[1], x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_matern.parameters):
            prod_term = self._K_from_prod(x1, x2, skip=[dim])  # N x M
            grad_term = self._dK_dx1_1d(x1[:, dim], x2[:, dim], kern.lengthscale[0])  # N x M
            dK_dx1[dim, :, :] *= prod_term * grad_term
        return dK_dx1


class BrownianGPy(IBrownian):
    r"""Wrapper of the GPy Brownian motion kernel as required for some EmuKit quadrature methods.

    .. math::
        k(x, x') = \sigma^2 \operatorname{min}(x, x')\quad\text{with}\quad x, x' \geq 0,

    where :math:`\sigma^2` is the ``variance`` property.

    :param gpy_brownian: A Brownian motion kernel from GPy.
    """

    def __init__(self, gpy_brownian: GPy.kern.Brownian):
        self.gpy_brownian = gpy_brownian

    @property
    def variance(self) -> float:
        return self.gpy_brownian.variance[0]

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_brownian.K(x1, x2)


class ProductBrownianGPy(IProductBrownian):
    r"""Wrapper of the GPy Brownian product kernel as required for some EmuKit quadrature methods.

    The product kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = \operatorname{min}(x_i-c, x_i'-c)\quad\text{with}\quad x_i, x_i' \geq c,

    :math:`d` is the input dimensionality,
    :math:`\sigma^2` is the ``variance`` property
    and :math:`c` is the ``offset`` property.

    :param gpy_brownian: A Brownian product kernel from GPy. For :math:`d=1` this is equivalent to a
                         Brownian kernel. For :math:`d>1`, this is a product of :math:`d` 1-dimensional Brownian
                         kernels with differing active dimensions constructed as k1 * k2 * ... .
                         Make sure to unlink all variances except the variance of the first kernel k1 in the product
                         as the variance of k1 will be used to represent :math:`\sigma^2`. If you are unsure what
                         to do, use the :attr:`input_dim` and :attr:`variance` parameter instead.
                         If :attr:`gpy_brownian` is not given, the :attr:`variance` and :attr:`input_dim`
                         argument is used.
    :param offset: The offset :math:`c` of the kernel. Defaults to 0.
    :param variance: The variance of the product kernel. Only used if :attr:`gpy_brownian` is not given. Defaults to 1.
    :param input_dim: The input dimension. Only used if :attr:`gpy_brownian` is not given.
    """

    def __init__(
        self,
        gpy_brownian: Optional[Union[GPy.kern.Brownian, GPy.kern.Prod]] = None,
        offset: float = 0.0,
        variance: Optional[float] = None,
        input_dim: Optional[int] = None,
    ):
        if gpy_brownian is not None:
            if input_dim is not None or variance is not None:
                warnings.warn("gpy_brownian and variance is given. The variance will be ignore.")
        else:
            if input_dim is None or variance is None:
                raise ValueError(
                    "Please provide a GPy product Brownian kernel or alternitvely the the variance and input_dim."
                )

        # default variance
        if variance is None:
            variance = 1.0

        # product kernel from parameters
        if gpy_brownian is None:
            gpy_brownian = GPy.kern.Brownian(input_dim=1, active_dims=[0], variance=variance)
            for dim in range(1, input_dim):
                k = GPy.kern.Brownian(input_dim=1, active_dims=[dim])
                k.unlink_parameter(k.variance)
                gpy_brownian = gpy_brownian * k

        self.gpy_brownian = gpy_brownian
        self._offset = offset

    @property
    def variance(self) -> float:
        if isinstance(self.gpy_brownian, GPy.kern.Brownian):
            return self.gpy_brownian.variance[0]

        return self.gpy_brownian.parameters[0].variance[0]

    @property
    def offset(self) -> float:
        return self._offset

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_brownian.K(x1 - self._offset, x2 - self.offset)

    def _K_from_prod(self, x1: np.ndarray, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        """The kernel k(x1, x2) with offset=0 evaluated at x1 and x2 computed as product from the
        individual 1d kernels.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :param skip: Skip these dimensions if specified.
        :returns: Kernel evaluated at x1, x2, shape (N, M).
        """
        if skip is None:
            skip = []
        K = np.ones([x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_brownian.parameters):
            if dim in skip:
                continue
            K *= kern.K(x1, x2)

        # correct for missing variance
        if 0 in skip:
            K *= self.variance
        return K

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if isinstance(self.gpy_brownian, GPy.kern.Brownian):
            return self._dK_dx1_1d(x1[:, 0], x2[:, 0])[None, :, :]

        # product kernel
        x1 = x1 - self.offset
        x2 = x2 - self.offset
        dK_dx1 = np.ones([x1.shape[1], x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_brownian.parameters):
            prod_term = self._K_from_prod(x1, x2, skip=[dim])  # N x M
            grad_term = self._dK_dx1_1d(x1[:, dim], x2[:, dim])  # N x M
            dK_dx1[dim, :, :] *= prod_term * grad_term
        return dK_dx1

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """The gradient of the diagonal of the kernel (the variance) v(x):=k(x, x) evaluated at x.

        :param x: The locations where the gradient is evaluated, shape (n_points, input_dim).
        :return: The gradient of the diagonal of the kernel evaluated at x, shape (input_dim, n_points).
        """
        if isinstance(self.gpy_brownian, GPy.kern.Brownian):
            return self.variance * np.ones((x.shape[1], x.shape[0]))

        x = x - self.offset
        dKdiag_dx = np.ones((x.shape[1], x.shape[0]))
        for dim, kern in enumerate(self.gpy_brownian.parameters):
            prod_term = np.prod(x, axis=1) / x[:, dim]  # N,
            grad_term = 1.0
            dKdiag_dx[dim, :] *= prod_term * grad_term
        return self.variance * dKdiag_dx


# === convenience functions start here


def create_emukit_model_from_gpy_model(
    gpy_model: GPy.models.GPRegression,
    integral_bounds: Optional[BoundsType] = None,
    measure: Optional[IntegrationMeasure] = None,
    integral_name: str = "",
) -> BaseGaussianProcessGPy:
    """Wraps a GPy model and returns an EmuKit quadrature model.

    :param gpy_model: A GPy Gaussian process regression model ``GPy.models.GPRegression``.
    :param integral_bounds: List of d tuples, where d is the dimensionality of the integral and the tuples contain the
                            lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_d, ub_d)].
                            Only used if ``measure`` is not given in which case the unnormalized Lebesgue measure is used.
    :param measure: An integration measure. Either ``measure`` or ``integral_bounds`` must be given.
                    If both ``integral_bounds`` and ``measure`` are given, ``integral_bounds`` is disregarded.
    :param integral_name: The (variable) name(s) of the integral.
    :return: An EmuKit GP model for quadrature with GPy backend.

    """

    if (integral_bounds is None) and (measure is None):
        raise ValueError("Either measure or integral bounds must be given.")

    if (integral_bounds is not None) and (measure is not None):
        warnings.warn("Both measure and integral bounds are given. Bounds are being ignored.")

    if measure is None:
        domain = BoxDomain(name="", bounds=integral_bounds)
        measure = LebesgueMeasure(domain=domain)

    def _check_is_gpy_product_kernel(k, k_type):
        is_type = isinstance(gpy_model.kern, k_type)
        if isinstance(k, GPy.kern.Prod):
            all_type = all(isinstance(kern, k_type) for kern in k.parameters)
            all_univariante = all(kern.input_dim == 1 for kern in k.parameters)
            if all_type and all_univariante:
                is_type = True
        return is_type

    # wrap standard kernel
    # RBF
    qkern_emukit = None
    if isinstance(gpy_model.kern, GPy.kern.RBF):
        skern_emukit = RBFGPy(gpy_model.kern)
        if isinstance(measure, LebesgueMeasure):
            qkern_emukit = QuadratureRBFLebesgueMeasure(skern_emukit, measure, integral_name)
        elif isinstance(measure, GaussianMeasure):
            qkern_emukit = QuadratureRBFGaussianMeasure(skern_emukit, measure, integral_name)

    # Univariate Matern12 or ProductMatern12
    elif _check_is_gpy_product_kernel(gpy_model.kern, GPy.kern.Exponential):
        skern_emukit = ProductMatern12GPy(gpy_model.kern)
        if isinstance(measure, LebesgueMeasure):
            qkern_emukit = QuadratureProductMatern12LebesgueMeasure(skern_emukit, measure, integral_name)

    # Univariate Matern32 or ProductMatern32
    elif _check_is_gpy_product_kernel(gpy_model.kern, GPy.kern.Matern32):
        skern_emukit = ProductMatern32GPy(gpy_model.kern)
        if isinstance(measure, LebesgueMeasure):
            qkern_emukit = QuadratureProductMatern32LebesgueMeasure(skern_emukit, measure, integral_name)

    # Univariate Matern52 or ProductMatern52
    elif _check_is_gpy_product_kernel(gpy_model.kern, GPy.kern.Matern52):
        skern_emukit = ProductMatern52GPy(gpy_model.kern)
        if isinstance(measure, LebesgueMeasure):
            qkern_emukit = QuadratureProductMatern52LebesgueMeasure(skern_emukit, measure, integral_name)

    # Brownian
    elif isinstance(gpy_model.kern, GPy.kern.Brownian):
        skern_emukit = BrownianGPy(gpy_model.kern)
        if isinstance(measure, LebesgueMeasure):
            qkern_emukit = QuadratureBrownianLebesgueMeasure(skern_emukit, measure, integral_name)

    # ProductBrownian
    elif _check_is_gpy_product_kernel(gpy_model.kern, GPy.kern.Brownian):
        skern_emukit = ProductBrownianGPy(gpy_model.kern)
        if isinstance(measure, LebesgueMeasure):
            qkern_emukit = QuadratureProductBrownianLebesgueMeasure(skern_emukit, measure, integral_name)

    else:
        raise ValueError(f"There is no GPy wrapper for the provided GPy kernel ({gpy_model.kern.name}).")

    if qkern_emukit is None:
        raise ValueError(f"Kernel embedding not available for provided kernel-measure combination.")

    # wrap the base-gp model
    return BaseGaussianProcessGPy(kern=qkern_emukit, gpy_model=gpy_model)
