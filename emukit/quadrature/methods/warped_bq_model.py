# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import numpy as np

from emukit.quadrature.measures import BoxDomain, IntegrationMeasure

from ...core.interfaces.models import IDifferentiable, IModel
from ...quadrature.interfaces.base_gp import IBaseGaussianProcess
from .warpings import Warping


class WarpedBayesianQuadratureModel(IModel, IDifferentiable):
    r"""The general class for Bayesian quadrature (BQ) with a warped Gaussian process model.

    The model is of the form :math:`f = w(g(x))` where :math:`g\sim\mathcal{GP}` is a Gaussian process
    and :math:`w:\mathbb{R}\rightarrow\mathbb{R}` is a deterministic warping function.

    Inherit from this class to create new warped Bayesian quadrature models.

    .. seealso::
        * :class:`emukit.quadrature.methods.warpings.Warping`
        * :class:`emukit.quadrature.methods.VanillaBayesianQuadrature`
        * :class:`emukit.quadrature.methods.BoundedBayesianQuadrature`
        * :class:`emukit.quadrature.methods.WSABIL`

    """

    def __init__(self, base_gp: IBaseGaussianProcess, warping: Warping, X: np.ndarray, Y: np.ndarray):
        """
        :param base_gp: The underlying Gaussian process model.
        :param warping: The warping of the underlying Gaussian process model.
        :param X: The initial locations of integrand evaluations, shape (n_points, input_dim).
        :param Y: The values of the integrand at X, shape (n_points, 1).
        """
        self._warping = warping
        self.base_gp = base_gp
        # set data to ensure that the base_gp get the correctly transformed observations.
        self.set_data(X, Y)

    @property
    def X(self) -> np.ndarray:
        """The data nodes."""
        return self.base_gp.X

    @property
    def Y(self) -> np.ndarray:
        """The data evaluations at the nodes."""
        return self._warping.transform(self.base_gp.Y)

    @property
    def integral_bounds(self) -> Union[None, BoxDomain]:
        """The integration bounds. ``None`` if integration domain is not bounded."""
        return self.base_gp.kern.integral_bounds

    @property
    def reasonable_box_bounds(self) -> BoxDomain:
        """Reasonable box bounds.

        This box is used by the acquisition optimizer even when ``integral_bounds`` is ``None``.
        By default it is set to :meth:`get_box()` of the integration measure used, or, if not available,
        to the ``integral_bounds``.

        .. seealso::
            :class:`emukit.quadrature.measures.IntegrationMeasure.get_box`

        """
        return self.base_gp.kern.reasonable_box

    @property
    def measure(self) -> Union[None, IntegrationMeasure]:
        """The measure used for integration. ``None`` for standard Lebesgue measure."""
        return self.base_gp.kern.measure

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """The transform from base-GP to integrand implicitly defined by the warping used."""
        return self._warping.transform(Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """The transform from integrand to base-GP implicitly defined by the warping used."""
        return self._warping.inverse_transform(Y)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute predictive means and variances of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim).
        :returns: Predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
                  all shapes (n_points, 1).
        """
        raise NotImplementedError

    def predict_base_with_full_covariance(
        self, X_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute predictive means and covariance of the warped GP as well as the base GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :returns: Predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
                  mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points)
        """
        raise NotImplementedError

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute predictive means and covariance of warped GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :return: predictive mean, predictive full covariance of warped-GP, shapes (n_points, 1) and (n_points, n_points)
        """
        return self.predict_base_with_full_covariance(X_pred)[:2]

    def predict(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute predictive means and variances of warped-GP.

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :return: predictive mean, predictive variances of warped-GP, both shapes (n_points, 1)
        """
        return self.predict_base(X_pred)[:2]

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Set the new data in the model.

        First, potential warping parameters that are not being optimized but do depend on the data
        in an analytic way are updated. This is done via the method :meth:`compute_warping_params`.
        Then, the new data is automatically transformed and set in the model.

        :param X: Observation locations, shape (n_points, input_dim).
        :param Y: Integrand observations at X, shape (n_points, 1).
        """
        self._warping.update_parameters(**self.compute_warping_params(X, Y))
        self.base_gp.set_data(X, self._warping.inverse_transform(Y))

    def compute_warping_params(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Compute new parameters of the warping that are dependent on data, and that are not being optimized.

        This method is called by default when new data is being set in :meth:`set_data`.
        By default, this method returns an empty dict (no warping params need to be updated).
        Override this method in case warping parameters are data dependent.

        .. seealso::
            :class:`emukit.quadrature.methods.warpings.Warping.update_parameters`

        :param X: Observation locations, shape (n_points, input_dim).
        :param Y: Integrand observations at X, shape (n_points, 1).
        :returns : Dictionary containing new warping parameters. Names of parameters are the keys.
        """
        return {}

    def optimize(self) -> None:
        """Optimizes the hyperparameters of the base GP."""
        self.base_gp.optimize()

    def integrate(self) -> Tuple[float, float]:
        """Compute an estimator of the integral as well as its variance.

        :returns: Estimator of integral and its variance.
        """
        raise NotImplementedError

    @staticmethod
    def symmetrize_matrix(A: np.ndarray) -> np.ndarray:
        r"""Symmetrize a matrix.

        The symmetrized matrix is computed as :math:`A_{sym} = \frac{1}{2} (A + A^{\intercal})`.

        :param A: The square matrix :math:`A`, shape (N, N)
        :return: The symmetrized matrix :math:`A_{sym}`, shape (N, N).
        """
        return 0.5 * (A + A.T)
