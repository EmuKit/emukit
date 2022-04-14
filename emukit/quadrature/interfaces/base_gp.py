# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from ...core.interfaces import IDifferentiable, IModel
from ..kernels.quadrature_kernels import QuadratureKernel


class IBaseGaussianProcess(IModel, IDifferentiable):
    """Interface for a Gaussian process as used by quadrature models.

    Implementations of this class can be used by Gaussian process based quadrature models
    such as :class:`WarpedBayesianQuadratureModel`.

    .. note::
        When this class is initialized with data, use the raw evaluations Y of the integrand
        as this is a general Gaussian process class that can be used with various quadrature methods.
        Possible transformations of Y such as the ones applied in :class:`WarpedBayesianQuadratureModel`
        will be performed automatically there.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IStandardKernel`
       * :class:`emukit.quadrature.methods.WarpedBayesianQuadratureModel`

    """

    def __init__(self, kern: QuadratureKernel) -> None:
        """
        :param kern: An instance of a quadrature kernel.
        """
        self.kern = kern

    @property
    def observation_noise_variance(self) -> np.float:
        """The variance of the Gaussian observation noise."""
        raise NotImplementedError

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean and full co-variance at the locations X_pred.

        :param X_pred: Points at which to predict, shape (n_points, input_dim)
        :return: Predictive mean, predictive full co-variance, shapes (num_points, 1) and (num_points, num_points).
        """
        raise NotImplementedError

    def solve_linear(self, z: np.ndarray) -> np.ndarray:
        """Solve the linear system :math:`Gx=z` for :math:`x`.

        :math:`G` is the Gram matrix :math:`G := K(X, X) + \sigma^2 I`,
        of shape (num_dat, num_dat) and :math:`z` is a matrix of shape (num_dat, num_obs).

        :param z: A matrix of shape (num_dat, num_obs).
        :return: The solution :math:`x` of the linear, shape (num_dat, num_obs).
        """
        raise NotImplementedError

    def graminv_residual(self) -> np.ndarray:
        r"""The solution :math:`z` of the linear system

        .. math::

            (K_{XX} + \sigma^2 I) z = (Y - m(X))

        where :math:`X` and :math:`Y` are the available nodes and function evaluation, :math:`m(X)`
        is the predictive mean at :math:`X`, and :math:`\sigma^2` the observation noise variance.

        :return: The solution :math:`z` of the linear system, shape (num_dat, 1).
        """
        raise NotImplementedError

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """Evaluates the predictive gradients of mean and variance at X.

        :param X: Points at which the gradients are evaluated, shape (n_points, input_dim).
        :returns: The gradients of mean and variance, each of shape (n_points, input_dim).
        """
        # gradient of mean
        d_mean_dx = (self.kern.dK_dx1(X, self.X) @ self.graminv_residual())[:, :, 0].T

        # gradient of variance
        dKdiag_dx = self.kern.dKdiag_dx(X)
        dKxX_dx1 = self.kern.dK_dx1(X, self.X)
        graminv_KXx = self.solve_linear(self.kern.K(self.X, X))
        d_var_dx = dKdiag_dx - 2.0 * (dKxX_dx1 * np.transpose(graminv_KXx)).sum(axis=2, keepdims=False)

        return d_mean_dx, d_var_dx.T
