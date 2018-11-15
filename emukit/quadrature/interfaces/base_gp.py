import numpy as np
from typing import Tuple

from emukit.core.interfaces import IModel
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel


class IBaseGaussianProcess(IModel):
    """ Interface with properties a GP model should have """

    def __init__(self, kern: QuadratureKernel) -> None:
        self.kern = kern

    @property
    def noise_variance(self):
        """
        Gaussian observation noise variance
        :return: The noise variance from some external GP model
        """
        raise NotImplementedError

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple:
        """
        Predictive mean and full co-variance at the locations X_pred

        :param X_pred: points at which to predict, with shape (number of points, dimension)
        :return: Predictive mean, predictive full co-variance
        """
        raise NotImplementedError

    def gram_chol(self) -> np.ndarray:
        """
        The lower triangular cholesky decomposition of the kernel Gram matrix

        :return: a lower triangular matrix being the cholesky matrix of the kernel Gram matrix
        """
        raise NotImplementedError

    def graminv_residual(self) -> np.ndarray:
        """
        The inverse Gram matrix multiplied with the mean-corrected data

        ..math::

            G_{XX}^{-1} (Y - m(X))

        where the data is given by {X, Y} and m is the prior mean

        :return: the inverse Gram matrix multiplied with the mean-corrected data with shape: (number of datapoints, 1)
        """
        raise NotImplementedError

