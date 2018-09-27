import numpy as np
from typing import Tuple


from emulab.quadrature.kernels import IntegrableKernel


class IBaseGaussianProcess(IModel, IGPQuantities):
    """
    Class to define the functionality of a GP class for base_gp required by the quadrature methods
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, kernel: IntegrableKernel) -> None:
        """
        :param X: locations of data
        :param Y: function evaluations at locations X
        :param kernel: integrable kernel
        :param noise_variance: Gaussian observation noise
        """
        self._X = X
        self._Y = Y
        self.kern = kernel

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def predict(self, X_pred: np.ndarray, full_cov: bool = False) -> Tuple:
        """
        Predictive mean and (co)variance at the locations X_pred

        :param X_pred: points at which to predict, with shape (number of points, dimension)
        :param full_cov: if True, return the full covariance matrix instead of just the variance
        :return: Predictive mean, predictive (co)variance
        """
        raise NotImplementedError


class IQuadratureBaseModel:
    """ Interface with properties a GP model should have """

    @property
    def noise_variance(self):
        """
        Gaussian observation noise variance
        :return: The noise variance from some external GP model
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
