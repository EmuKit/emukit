# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List

from emukit.quadrature.interfaces.standard_kernels import IRBF
from emukit.quadrature.interfaces import IBaseGaussianProcess
from emukit.quadrature.kernels.integral_bounds import IntegralBounds
from emukit.quadrature.interfaces import IStandardKernel


class BaseGaussianProcessGPy(IBaseGaussianProcess):
    """
    Wrapper for GPy GPRegression

    An instance of this can be passed as 'base_gp' to an ApproximateWarpedGPSurrogate object
    """
    def __init__(self, standard_kern: IStandardKernel, gpy_model, integral_bounds: IntegralBounds):
        """
        :param standard_kern: a standard kernel
        :param gpy_model: A GPy GP regression model, GPy.models.GPRegression
        """
        super().__init__(standard_kern=standard_kern, integral_bounds=integral_bounds)
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

    def predict(self, X_pred: np.ndarray, full_cov=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predictive mean and (co)variance at the locations X_pred

        :param X_pred: points at which to predict, with shape (number of points, dimension)
        :param full_cov: if True, return the full covariance matrix instead of just the variance
        :return: Predictive mean, predictive (co)variance
        """
        return self.gpy_model.predict(X_pred, full_cov)

    def gram_chol(self) -> np.ndarray:
        """
        The lower triangular cholesky decomposition of the kernel Gram matrix

        :return: a lower triangular matrix being the cholesky matrix of the kernel Gram matrix
        """
        return self.gpy_model.posterior.woodbury_chol

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
        """ Optimize the hyperparameters of the model """
        self.gpy_model.optimize()


class RBFGPy(IRBF):
    """
    Wrapper of the GPy RBF kernel

    .. math::
        k(x, x') = \sigma^2 e^{-\frac{1}{2}\frac{\|x-x'\|^2}{\lambda^2}},

    where :math:`\sigma^2` is the `variance' property and :math:`\lambda` is the lengthscale property.
    """

    def __init__(self, gpy_rbf):
        """
        :param gpy_rbf: An RBF kernel from GPy
        """
        self.gpy_rbf = gpy_rbf

    @property
    def lengthscale(self) -> np.float:
        return self.gpy_rbf.lengthscale

    @property
    def variance(self) -> np.float:
        return self.gpy_rbf.variance

    def K(self, x, x2=None) -> np.ndarray:
        """
        The kernel evaluated at x and x2

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (N, M)
        """
        return self.gpy_rbf.K(x, x2)

    def dK_dx(self, x, x2) -> np.ndarray:
        """
        gradient of the kernel wrt x

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        return self.gpy_rbf.dK_dr_via_X(x, x2)[None, ...] * self._dr_dx(x, x2)

    # helper
    def _dr_dx(self, x, x2) -> np.ndarray:
        """
        Derivative of the radius

        .. math::

            r = \sqrt{ \frac{||x - x_2||^2}{\lambda^2} }

        name mapping:
            \lambda: self.rbf.lengthscale

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        return (x.T[:, :, None] - x2.T[:, None, :]) / \
               (self.lengthscale ** 2 * (x.T[:, :, None] - x2.T[:, None, :]) / (self.lengthscale * np.sqrt(2)))


def convert_gpy_model_to_emukit_model(gpy_model, integral_bounds: List, integral_name: str='') \
        -> BaseGaussianProcessGPy:
    """
    Wraps a GPy model and returns an emukit quadrature model

    :param gpy_model: A GPy Gaussian process regression model, GPy.models.GPRegression
    :param integral_bounds: List of D tuples, where D is the dimensionality of the integral and the tuples contain the
        lower and upper bounds of the integral i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
    :param integral_name: the (variable) name(s) of the integral
    (lower bounds of dimension i, upper bound of dimension i) for i=1,...,integral dim.

    :return: emukit model for quadrature witg GPy backend (IBaseGaussianProcessGPy)
    """

    # get the integral bounds
    bounds = IntegralBounds(name=integral_name, bounds=integral_bounds)

    # warp the standard kernel
    if gpy_model.kern.name is 'rbf':
        standard_kernel = RBFGPy(gpy_model.kern)
    else:
        raise NotImplementedError("Only the GPy rbf-kernel is supported right now.")

    # wrap the model
    return BaseGaussianProcessGPy(standard_kern=standard_kernel, gpy_model=gpy_model, integral_bounds=bounds)
