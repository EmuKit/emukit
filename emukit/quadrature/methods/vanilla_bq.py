# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from ...quadrature.interfaces.base_gp import IBaseGaussianProcess
from .warped_bq_model import WarpedBayesianQuadratureModel
from .warpings import IdentityWarping


class VanillaBayesianQuadrature(WarpedBayesianQuadratureModel):
    """Standard ('vanilla') Bayesian quadrature.

    The warping for vanilla Bayesian quadrature is the identity transform :math:`w(y) = y`.
    Hence, the model for the integrand :math:`f` is a standard Gaussian process as.

    :param base_gp: The underlying Gaussian process model.
    :param X: The initial locations of integrand evaluations, shape (n_points, input_dim).
    :param Y: The values of the integrand at X, shape (n_points, 1).

    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray):
        super(VanillaBayesianQuadrature, self).__init__(base_gp=base_gp, warping=IdentityWarping(), X=X, Y=Y)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m, cov = self.base_gp.predict(X_pred)
        return m, cov, m, cov

    def predict_base_with_full_covariance(
        self, X_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m, cov = self.base_gp.predict_with_full_covariance(X_pred)
        return m, cov, m, cov

    def integrate(self) -> Tuple[float, float]:
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, 0]
        integral_var = self.base_gp.kern.qKq() - (kernel_mean_X @ self.base_gp.solve_linear(kernel_mean_X.T))[0, 0]
        return integral_mean, integral_var

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        return self.base_gp.get_prediction_gradients(X)
