# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Tuple, Union

import GPy
import numpy as np
from scipy.linalg import lapack

from emukit.quadrature.interfaces import IBaseGaussianProcess
from emukit.quadrature.interfaces.standard_kernels import IRBF, IProductMatern32
from emukit.quadrature.measures import IntegrationMeasure, IsotropicGaussianMeasure, UniformMeasure
from emukit.quadrature.kernels import QuadratureKernel, QuadratureProductMatern32LebesgueMeasure, QuadratureRBFIsoGaussMeasure, QuadratureRBFLebesgueMeasure, QuadratureRBFUniformMeasure


class BaseGaussianProcessGPy(IBaseGaussianProcess):
    """Wrapper for GPy GPRegression.

    An instance of this can be passed as 'base_gp' to a WarpedBayesianQuadratureModel object.

    Note that the GPy cannot take None as initial values for X and Y. Thus we initialize it with some values. These will
    be re-set in the WarpedBayesianQuadratureModel.
    """

    def __init__(self, kern: QuadratureKernel, gpy_model: GPy.models.GPRegression, noise_free: bool = True):
        """
        :param kern: A quadrature kernel.
        :param gpy_model: A GPy GP regression model.
        :param noise_free: If False, the observation noise variance will be treated as a model parameter,
                           if True it is set to 1e-10, defaults to True.
        """
        super().__init__(kern=kern)
        if noise_free:
            gpy_model.Gaussian_noise.constrain_fixed(1.0e-10)
        self.gpy_model = gpy_model

    @property
    def X(self) -> np.ndarray:
        return self.gpy_model.X

    @property
    def Y(self) -> np.ndarray:
        return self.gpy_model.Y

    @property
    def observation_noise_variance(self) -> np.float:
        """Gaussian observation noise variance.

        :return: The noise variance from some external GP model
        """
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
        """Solve the linear system :math:`G(X, X)x=z` for :math:`x`.

        :math:`G(X, X)` is the Gram matrix :math:`G(X, X) = K(X, X) + \sigma^2 I`, of shape (num_dat, num_dat)
        and :math:`z` is a matrix of shape (num_dat, num_points).

        :param z: A matrix of shape (num_dat, num_obs).
        :return: The solution :math:`x` of linear system, shape (num_dat, num_points).
        """
        lower_chol = self.gpy_model.posterior.woodbury_chol
        return lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, z, lower=1)[0]), lower=0)[0]

    def graminv_residual(self) -> np.ndarray:
        """The inverse Gram matrix multiplied with the mean-corrected data.

        ..math::

            (K_{XX} + \sigma^2 I)^{-1} (Y - m(X))

        where the data is given by {X, Y} and m is the prior mean and :math:`sigma^2` the observation noise.

        :return: The inverse Gram matrix multiplied with the mean-corrected, shape: (num_points, 1).
        """
        return self.gpy_model.posterior.woodbury_vector

    def optimize(self) -> None:
        """Optimize the hyperparameters of the GP"""
        self.gpy_model.optimize()


class RBFGPy(IRBF):
    """Wrapper of the GPy RBF kernel.

    .. math::
        k(x, x') = \sigma^2 e^{-\frac{1}{2}\frac{\|x-x'\|^2}{\lambda^2}},

    where :math:`\sigma^2` is the `variance' property and :math:`\lambda` is the lengthscale property.
    """

    def __init__(self, gpy_rbf: GPy.kern.RBF):
        """
        :param gpy_rbf: An RBF kernel from GPy with ARD=False
        """
        if gpy_rbf.ARD:
            raise ValueError("ARD of the GPy kernel must be set to False.")
        self.gpy_rbf = gpy_rbf

    @property
    def lengthscale(self) -> np.float:
        return self.gpy_rbf.lengthscale[0]

    @property
    def variance(self) -> np.float:
        return self.gpy_rbf.variance.values[0]

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2.

        :param x1: First argument of the kernel.
        :param x2: Second argument of the kernel.
        :returns: Kernel evaluated at x1, x2.
        """
        return self.gpy_rbf.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Gradient of the kernel wrt x1 evaluated at pair x1, x2.

        :param x1: First argument of the kernel, shape = (n_points N, input_dim).
        :param x2: Second argument of the kernel, shape = (n_points M, input_dim).
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M).
        """
        K = self.K(x1, x2)
        scaled_vector_diff = (x1.T[:, :, None] - x2.T[:, None, :]) / self.lengthscale**2
        dK_dx1 = -K[None, ...] * scaled_vector_diff
        return dK_dx1


class ProductMatern32GPy(IProductMatern32):
    """Wrapper of the GPy Matern32 product kernel.

    The ProductMatern32 kernel is of the form
    :math:`k(x, x') = \sigma^2 \prod_{i=1}^d k_i(x, x')` where

    .. math::
        k_i(x, x') = (1 + \sqrt{3}r_i ) e^{-\sqrt{3} r_i}.

    :math:`d` is the input dimensionality, :math:`r_i:=\frac{|x_i - z_i|}{\lambda_i}}`,
    :math:`\sigma^2` is the ``variance`` property and :math:`\lambda_i` is the ith element
    of the ``lengthscales`` property.
    """

    def __init__(
        self,
        gpy_matern: Optional[Union[GPy.kern.Matern32, GPy.kern.Prod]] = None,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[float] = None,
    ):
        """
        :param gpy_matern: An Matern32 (product) kernel from GPy. For d > 1, this is not a d-dimensional Matern32 kernel
                           but a product of d 1-dimensional Matern32 kernels with differing active dimensions
                           constructed as k1 * k2 * ... .
                           Make sure to unlink all variances except the variance of the first kernel k1 in the product
                           as the variance of k1 will be used to represent :math:`sigma^2`.
                           If ``gpy_matern`` is not given, the ``lengthscales`` argument is used.
        :param lengthscales: If ``gpy_matern`` is not given, a product Matern32 kernel will be constructed with
                           the given lengthscales. The number of elements need to be equal to the dimensionality
                           d. If ``gpy_matern`` is given, this input is disregarded.
        :param variance: The variance of the product kernel. Only used if ``gpy_matern`` is not given. Defaults to 1.
        """
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
    def variance(self) -> np.float:
        if isinstance(self.gpy_matern, GPy.kern.Matern32):
            return self.gpy_matern.variance[0]

        return self.gpy_matern.parameters[0].variance[0]

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2.

        :param x1: First argument of the kernel.
        :param x2: Second argument of the kernel.
        :returns: Kernel evaluated at x1, x2.
        """
        return self.gpy_matern.K(x1, x2)

    def _K_from_prod(self, x1: np.ndarray, x2: np.ndarray, skip: List[int] = None) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2 computed as product from the
        individual 1d kernels.

        :param x1: First argument of the kernel.
        :param x2: Second argument of the kernel.
        :param skip: Skip these dimensions if specified.
        :returns: Kernel evaluated at x1, x2.
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
        """Gradient of the kernel wrt x1 evaluated at pair x1, x2.

        :param x1: First argument of the kernel, shape = (n_points N, input_dim).
        :param x2: Second argument of the kernel, shape = (n_points M, input_dim).
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M).
        """

        if isinstance(self.gpy_matern, GPy.kern.Matern32):
            return self._dK_dx_1d(x1[:, 0], x2[:, 0], self.gpy_matern)[None, :, :]

        # product kernel
        dK_dx1 = np.ones([x1.shape[1], x1.shape[0], x2.shape[0]])
        for dim, kern in enumerate(self.gpy_matern.parameters):
            prod_term = self._K_from_prod(x1, x2, skip=[dim])  # N x M
            grad_term = self._dK_dx_1d(x1[:, dim], x2[:, dim], kern)  # N x M
            dK_dx1[dim, :, :] *= prod_term * grad_term
        return dK_dx1

    def _dK_dx_1d(self, x1: np.ndarray, x2: np.ndarray, kern: GPy.kern.Matern32) -> np.ndarray:
        r = (x1.T[:, None] - x2.T[None, :]) / kern.lengthscale[0]  # N x M
        dr_dx1 = r / (kern.lengthscale[0] * abs(r))
        dK_dr = -3 * abs(r) * np.exp(-np.sqrt(3) * abs(r))
        return dK_dr * dr_dx1


def create_emukit_model_from_gpy_model(
    gpy_model: GPy.models.GPRegression,
    integral_bounds: Optional[List[Tuple[float, float]]],
    measure: Optional[IntegrationMeasure],
    integral_name: str = "",
) -> BaseGaussianProcessGPy:
    """Wraps a GPy model and returns an emukit quadrature model.

    :param gpy_model: A GPy Gaussian process regression model, GPy.models.GPRegression.
    :param integral_bounds: List of D tuples, where D is the dimensionality of the integral and the tuples contain the
                            lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)].
                            None means infinite bounds.
    :param measure: an integration measure. None means the standard Lebesgue measure is used.
    :param integral_name: the (variable) name(s) of the integral.
    :return: emukit model for quadrature with GPy backend (IBaseGaussianProcessGPy)
    """

    # neither measure nor bounds are given
    if (integral_bounds is None) and (measure is None):
        raise ValueError(
            "Integral_bounds are infinite and measure is standard Lebesgue. Choose either finite bounds "
            "or an appropriate integration measure."
        )

    def _check_is_product_matern32(k):
        is_matern = isinstance(gpy_model.kern, GPy.kern.Matern32)
        if isinstance(k, GPy.kern.Prod):
            all_matern = all(isinstance(kern, GPy.kern.Matern32) for kern in k.parameters)
            all_univariante = all(kern.input_dim == 1 for kern in k.parameters)
            if all_matern and all_univariante:
                is_matern = True
        return is_matern

    # wrap standard kernel
    # RBF
    if isinstance(gpy_model.kern, GPy.kern.RBF):
        standard_kernel_emukit = RBFGPy(gpy_model.kern)
        quadrature_kernel_emukit = _get_qkernel_gauss(standard_kernel_emukit, integral_bounds, measure, integral_name)
    # Univariate Matern32 or ProductMatern32
    elif _check_is_product_matern32(gpy_model.kern):
        standard_kernel_emukit = ProductMatern32GPy(gpy_model.kern)
        quadrature_kernel_emukit = _get_qkernel_matern32(
            standard_kernel_emukit, integral_bounds, measure, integral_name
        )
    else:
        raise ValueError("Only RBF and ProductMatern32 kernel are supported. Got ", gpy_model.kern.name, " instead.")

    # wrap the base-gp model
    return BaseGaussianProcessGPy(kern=quadrature_kernel_emukit, gpy_model=gpy_model)


def _get_qkernel_matern32(
    standard_kernel_emukit: IProductMatern32,
    integral_bounds: Optional[List[Tuple[float, float]]],
    measure: Optional[IntegrationMeasure],
    integral_name: str,
):
    # we already know that either bounds or measure is given (or both)
    # finite bounds, standard Lebesgue measure
    if (integral_bounds is not None) and (measure is None):
        quadrature_kernel_emukit = QuadratureProductMatern32LebesgueMeasure(
            matern_kernel=standard_kernel_emukit, integral_bounds=integral_bounds, variable_names=integral_name
        )

    else:
        raise ValueError("Currently only standard Lebesgue measure (measure=None) is supported.")

    return quadrature_kernel_emukit


def _get_qkernel_gauss(
    standard_kernel_emukit: IRBF,
    integral_bounds: Optional[List[Tuple[float, float]]],
    measure: Optional[IntegrationMeasure],
    integral_name: str,
):
    # we already know that either bounds or measure is given (or both)
    # infinite bounds: Gauss or uniform measure only
    if (integral_bounds is None) and (measure is not None):
        if isinstance(measure, UniformMeasure):
            quadrature_kernel_emukit = QuadratureRBFUniformMeasure(
                rbf_kernel=standard_kernel_emukit,
                integral_bounds=integral_bounds,
                measure=measure,
                variable_names=integral_name,
            )
        elif isinstance(measure, IsotropicGaussianMeasure):
            quadrature_kernel_emukit = QuadratureRBFIsoGaussMeasure(
                rbf_kernel=standard_kernel_emukit, measure=measure, variable_names=integral_name
            )
        else:
            raise ValueError(
                "Currently only IsotropicGaussianMeasure and UniformMeasure supported with infinite integral bounds."
            )

    # finite bounds, standard Lebesgue measure
    elif (integral_bounds is not None) and (measure is None):
        quadrature_kernel_emukit = QuadratureRBFLebesgueMeasure(
            rbf_kernel=standard_kernel_emukit, integral_bounds=integral_bounds, variable_names=integral_name
        )

    # finite bounds: measure: uniform measure only
    else:
        if isinstance(measure, UniformMeasure):
            quadrature_kernel_emukit = QuadratureRBFUniformMeasure(
                rbf_kernel=standard_kernel_emukit,
                integral_bounds=integral_bounds,
                measure=measure,
                variable_names=integral_name,
            )
        else:
            raise ValueError(
                "Currently only standard Lebesgue measure (measure=None) is supported with finite integral bounds."
            )

    return quadrature_kernel_emukit
