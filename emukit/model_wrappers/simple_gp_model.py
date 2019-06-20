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

    def __repr__(self):
        """
        Prints the hyper-parameters
        """
        return 'Lengthscale: {:.4f} \n Kernel variance: {:.4f} \n Likelihood variance: {:.4f}'.format(
            self.lengthscale,
            self.kernel_variance,
            self.likelihood_variance)

    def optimize(self) -> None:
        """
        Optimize the three hyperparameters of the model, namely the kernel variance, kernel lengthscale and likelihood
        variance
        """
        def optimize_fcn(log_hyper_parameters):
            # take exponential to ensure positive values
            hyper_parameters = np.exp(log_hyper_parameters)
            self.lengthscale = hyper_parameters[0]
            self.kernel_variance = hyper_parameters[1]
            self.likelihood_variance = hyper_parameters[2]
            return self._negative_marginal_log_likelihood()

        lower_bound = np.log(1e-6)
        upper_bound = np.log(1e8)

        bounds = [(lower_bound, upper_bound) for _ in range(3)]
        scipy.optimize.minimize(optimize_fcn, np.log(np.array([self.lengthscale,
                                                               self.kernel_variance,
                                                               self.likelihood_variance])), bounds=bounds)

    def predict(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict from model

        :param x_new: (n_points, n_dims) array containing points at which the predictive distributions will be computed
        :return: Tuple containing two (n_points, 1) arrays representing the mean and variance of the predictive
                 distribution at the specified input locations
        """
        K = self._calc_kernel(self.x)
        K += np.identity(self.x.shape[0]) * (self.jitter + self.likelihood_variance)

        L = np.linalg.cholesky(K)

        K_xs = self._calc_kernel(self.x, x_new)

        tmp = scipy.linalg.solve_triangular(L, K_xs, lower=True)
        tmp2 = scipy.linalg.solve_triangular(L, self.y, lower=True)

        mean = np.dot(tmp.T, tmp2)
        variance = (self.kernel_variance - np.sum(np.square(tmp), axis=0))[:, None]
        return mean, variance

    def _calc_kernel(self, X: np.ndarray, X2: np.ndarray=None) -> np.ndarray:
        """
        Implements an RBF kernel with no ARD

        :param X: array of shape (n_points_1, n_dims) containing input points of first argument to kernel function
        :param X2: array of shape (n_points_2, n_dims) containing input points of second argument to kernel function.
                   If not supplied K(X, X) is computed.
        :return: Kernel matrix K(X, X2) or K(X, X) if X2 not supplied.
        """
        if X2 is None:
            X2 = X

        X1sq = np.sum(np.square(X), 1)
        X2sq = np.sum(np.square(X2), 1)
        r2 = -2. * np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
        r2 = np.clip(r2, 0, np.inf)
        return self.kernel_variance * np.exp(-0.5 * r2 / self.lengthscale ** 2)

    def _negative_marginal_log_likelihood(self) -> float:
        """
        :return: Negative marginal log likelihood of model with current hyper-parameters
        """
        K = self._calc_kernel(self.x)

        # Add some jitter to the diagonal
        K += np.identity(self.x.shape[0]) * (self.jitter + self.likelihood_variance)

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
