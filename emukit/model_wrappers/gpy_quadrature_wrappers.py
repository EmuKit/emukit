# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Tuple

import GPy
import numpy as np
from scipy.linalg import lapack

from emukit.quadrature.interfaces import IBaseGaussianProcess
from emukit.quadrature.interfaces.standard_kernels import IRBF, IMatern32
from emukit.quadrature.kernels.integration_measures import IntegrationMeasure, IsotropicGaussianMeasure, UniformMeasure
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel
from emukit.quadrature.kernels.quadrature_rbf import (
    QuadratureRBFIsoGaussMeasure,
    QuadratureRBFLebesgueMeasure,
    QuadratureRBFUniformMeasure,
)


class BaseGaussianProcessGPy(IBaseGaussianProcess):
    """
    Wrapper for GPy GPRegression
    An instance of this can be passed as 'base_gp' to a WarpedBayesianQuadratureModel object.

    Note that the GPy cannot take None as initial values for X and Y. Thus we initialize it with some values. These will
    be re-set in the WarpedBayesianQuadratureModel.
    """

    def __init__(self, kern: QuadratureKernel, gpy_model: GPy.models.GPRegression, noise_free: bool = True):
        """
        :param kern: a quadrature kernel
        :param gpy_model: A GPy GP regression model, GPy.models.GPRegression
        :param noise_free: if False, the observation noise variance will be treated as a model parameter,
        if True it is set to 1e-10, defaults to True
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
        """
        Gaussian observation noise variance
        :return: The noise variance from some external GP model
        """
        return self.gpy_model.Gaussian_noise[0]

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model
        :param X: New training features
        :param Y: New training outputs
        """
        self.gpy_model.set_XY(X, Y)

    def predict(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predictive mean and covariance at the locations X_pred

        :param X_pred: points at which to predict, with shape (number of points, dimension)
        :return: Predictive mean, predictive variances shapes (num_points, 1) and (num_points, 1)
        """
        return self.gpy_model.predict(X_pred, full_cov=False)

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predictive mean and covariance at the locations X_pred

        :param X_pred: points at which to predict, with shape (num_points, input_dim)
        :return: Predictive mean, predictive full covariance shapes (num_points, 1) and (num_points, num_points)
        """
        return self.gpy_model.predict(X_pred, full_cov=True)

    def solve_linear(self, z: np.ndarray) -> np.ndarray:
        """
        Solve the linear system G(X, X)x=z for x.
        G(X, X) is the Gram matrix :math:`G(X, X) = K(X, X) + \sigma^2 I`, of shape (num_dat, num_dat) and z is a
        matrix of shape (num_dat, num_obs).

        :param z: a matrix of shape (num_dat, num_obs)
        :return: the solution to the linear system G(X, X)x = z, shape (num_dat, num_obs)
        """
        lower_chol = self.gpy_model.posterior.woodbury_chol
        return lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, z, lower=1)[0]), lower=0)[0]

    def graminv_residual(self) -> np.ndarray:
        """
        The inverse Gram matrix multiplied with the mean-corrected data

        ..math::

            (K_{XX} + \sigma^2 I)^{-1} (Y - m(X))

        where the data is given by {X, Y} and m is the prior mean and sigma^2 the observation noise

        :return: the inverse Gram matrix multiplied with the mean-corrected data with shape: (number of datapoints, 1)
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
        """
        The kernel k(x1, x2) evaluated at x1 and x2

        :param x1: first argument of the kernel
        :param x2: second argument of the kernel
        :returns: kernel evaluated at x1, x2
        """
        return self.gpy_rbf.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel wrt x1 evaluated at pair x1, x2.
        We use the scaled squared distance defined as:

        ..math::

            r^2(x_1, x_2) = \sum_{d=1}^D (x_1^d - x_2^d)^2/\lambda^2

        :param x1: first argument of the kernel, shape = (n_points N, input_dim)
        :param x2: second argument of the kernel, shape = (n_points M, input_dim)
        :return: the gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M)
        """
        K = self.K(x1, x2)
        scaled_vector_diff = (x1.T[:, :, None] - x2.T[:, None, :]) / self.lengthscale**2
        dK_dx1 = -K[None, ...] * scaled_vector_diff
        return dK_dx1


class Matern32GPy(IMatern32):
    """Wrapper of the GPy Matern kernel.

    .. math::
        k(x, x') = \sigma^2 (1 + \sqrt{3}r ) e^{-\sqrt{3}r},

    where :math:`r:=\sqrt{\sum_{i=1}^d\frac{(x_i - z_i)^2}{\lambda_i^2}}` and
    :math:`\sigma^2` is the `variance' property and :math:`\lambda` is the lengthscale property.
    """

    def __init__(self, gpy_matern: GPy.kern.Matern32):
        """
        :param gpy_matern: An Matern32 kernel from GPy.
        """
        self.gpy_matern = gpy_matern

    @property
    def lengthscales(self) -> np.ndarray:
        if not self.ARD:
            return np.full((self.gpy_matern.input_dim,), self.gpy_matern.lengthscale[0])
        return self.gpy_matern.lengthscale.values

    @property
    def variance(self) -> np.float:
        return self.gpy_matern.variance[0]

    @property
    def ARD(self) -> bool:
        return self.gpy_matern.ARD

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        The kernel k(x1, x2) evaluated at x1 and x2

        :param x1: first argument of the kernel
        :param x2: second argument of the kernel
        :returns: kernel evaluated at x1, x2
        """
        return self.gpy_matern.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Gradient of the kernel wrt x1 evaluated at pair x1, x2.
        We use the scaled squared distance defined as:

        ..math::

            r^2(x_1, x_2) = \sum_{d=1}^D (x_1^d - x_2^d)^2/\lambda^2

        :param x1: First argument of the kernel, shape = (n_points N, input_dim).
        :param x2: Second argument of the kernel, shape = (n_points M, input_dim).
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M).
        """
        r = x1.T[:, :, None] - x2.T[:, None, :]
        dr_dx1 = 1.0 * r
        if self.ARD:
            for d, ell in enumerate(self.lengthscales):
                r[d, :, :] = r[d, :, :] ** 2 / self.lengthscales[d] ** 2
                dr_dx1[d, :, :] /= self.lengthscales[d] ** 2
        else:
            r = r ** 2 / self.lengthscales ** 2
            dr_dx1 /= self.lengthscales[0] ** 2

        r = np.sqrt(r.sum(axis=0))

        K = self.K(x1, x2)
        first_term = self.variance * np.exp(-np.sqrt(3) * r)
        first_term = first_term[None, ...] * dr_dx1
        second_term = - np.sqrt(3) * (K[None, ...] * dr_dx1)
        return first_term + second_term


def create_emukit_model_from_gpy_model(
    gpy_model: GPy.models.GPRegression,
    integral_bounds: Optional[List[Tuple[float, float]]],
    measure: Optional[IntegrationMeasure],
    integral_name: str = "",
) -> BaseGaussianProcessGPy:
    """
    Wraps a GPy model and returns an emukit quadrature model.

    :param gpy_model: A GPy Gaussian process regression model, GPy.models.GPRegression
    :param integral_bounds: List of D tuples, where D is the dimensionality of the integral and the tuples contain the
    lower and upper bounds of the integral i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]. None means infinite
    bounds.
    :param measure: an integration measure. None means the standard Lebesgue measure is used.
    :param integral_name: the (variable) name(s) of the integral

    :return: emukit model for quadrature with GPy backend (IBaseGaussianProcessGPy)
    """

    # neither measure nor bounds are given
    if (integral_bounds is None) and (measure is None):
        raise ValueError(
            "Integral_bounds are infinite and measure is standard Lebesgue. Choose either finite bounds "
            "or an appropriate integration measure."
        )

    # wrap standard kernel
    if isinstance(gpy_model.kern, GPy.kern.RBF):
        standard_kernel_emukit = RBFGPy(gpy_model.kern)
        quadrature_kernel_emukit = _get_qkernel_gauss(standard_kernel_emukit, integral_bounds, measure, integral_name)
    elif isinstance(gpy_model.kern, GPy.kern.Matern32):
        standard_kernel_emukit = Matern32GPy(gpy_model.kern)
    else:
        raise ValueError("Only RBF and Matern32 kernel are supported. Got ", gpy_model.kern.name, " instead.")


    # wrap the base-gp model
    return BaseGaussianProcessGPy(kern=quadrature_kernel_emukit, gpy_model=gpy_model)


def _get_qkernel_matern32(
    standard_kernel_emukit: IRBF,
    integral_bounds: Optional[List[Tuple[float, float]]],
    measure: Optional[IntegrationMeasure],
    integral_name: str,
):
    # we already know that either bounds or measure is given (or both)
    # infinite bounds: Gauss or uniform measure only

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
                "Currently only IsotropicGaussianMeasure and UniformMeasure supported with infinite " "integral bounds."
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
