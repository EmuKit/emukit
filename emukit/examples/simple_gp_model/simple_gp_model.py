from typing import Tuple

import numpy as np
import scipy.linalg
import scipy.optimize

from emukit.core.interfaces import IModel


class SimpleGaussianProcessModel(IModel):
    """
    This model is a Gaussian process with an RBF kernel, with no ARD. It is used to demonstrate uses of emukit,
    it does not aim to be flexible, robust or fast.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        :param x: (n_points, n_dims) array containing training features
        :param y: (n_points, 1) array containing training targets
        """
        self.x = x
        self.y = y
        self.lengthscale = 1
        self.kernel_variance = 1
        self.likelihood_variance = 1
        self.jitter = 1e-6

    def optimize(self) -> None:
        """
        Optimize the three hyperparameters of the model, namely the kernel variance, kernel lengthscale and likelihood
        variance
        """
        def optimize_fcn(x):
            # take exponential to ensure positive values
            x = np.exp(x)
            self.lengthscale = x[0]
            self.kernel_variance = x[1]
            self.likelihood_variance = x[2]
            return self._negative_marginal_log_likelihood()

        scipy.optimize.minimize(optimize_fcn, np.log(np.array([self.lengthscale,
                                                               self.kernel_variance,
                                                               self.likelihood_variance])))

    def predict(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict from model

        :param x_new: (n_points, n_dims) array containing points at which the predictive distributions will be computed
        :return: Tuple containing two (n_points, 1) arrays representing the mean and variance of the predictive
                 distribution at the specified input locations
        """
        K = self._calc_kernel(self.x)
        idxs = list(range(self.x.shape[0]))
        K[idxs, idxs] += self.jitter + self.likelihood_variance
        L = np.linalg.cholesky(K)

        K_xs = self._calc_kernel(self.x, x_new)

        tmp = scipy.linalg.solve_triangular(L, K_xs, lower=True)
        tmp2 = scipy.linalg.solve_triangular(L, self.y, lower=True)

        mu = np.dot(tmp.T, tmp2)
        var = (self.kernel_variance - np.sum(np.square(tmp), axis=0))[:, None]
        return mu, var

    def _calc_kernel(self, X, X2=None):
        """
        Implements an RBF kernel with no ARD
        """
        if X2 is None:
            Xsq = np.sum(np.square(X), 1)

            XXt = np.dot(X, X.T)
            r2 = -2. * XXt + (Xsq[:, None] + Xsq[None, :])
            r2 = np.clip(r2, 0, np.inf)
        else:
            X1sq = np.sum(np.square(X), 1)
            X2sq = np.sum(np.square(X2), 1)
            r2 = -2. * np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
            r2 = np.clip(r2, 0, np.inf)

        return self.kernel_variance * np.exp(-r2 / self.lengthscale ** 2)

    def _negative_marginal_log_likelihood(self) -> float:
        """
        Negative marginal log likelihood
        """
        K = self._calc_kernel(self.x)

        # Add some jitter to the diagonal
        idxs = list(range(self.x.shape[0]))
        K[idxs, idxs] += self.jitter + self.likelihood_variance

        # cholesky decomposition of covariance matrix
        L = np.linalg.cholesky(K)

        # Log determinant of the covariance matrix
        log_det = 2. * np.sum(np.log(np.diag(L)))

        # calculate y^T K^{-1} y using the cholesky of K
        tmp = scipy.linalg.solve_triangular(L, self.y, lower=True)
        alpha = scipy.linalg.solve_triangular(L.T, tmp, lower=False)
        log_2_pi = np.log(2 * np.pi)
        return -0.5 * (-self.y.size * log_2_pi - self.y.shape[1] * log_det - np.sum(alpha * self.y))

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Set training data to new values
        :param X: (n_points, n_dims) array containing training features
        :param Y: (n_points, 1) array containing training targets
        """
        self.x = X
        self.y = Y

    @property
    def X(self) -> np.ndarray:
        return self.x

    @property
    def Y(self) -> np.ndarray:
        return self.y
