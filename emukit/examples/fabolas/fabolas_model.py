from copy import deepcopy
from typing import Tuple

import GPy
import numpy as np

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper


class FabolasKernel(GPy.kern.Kern):

    def __init__(self, input_dim, basis_func, a=1., b=1., active_dims=None):

        super(FabolasKernel, self).__init__(input_dim, active_dims, "fabolas_kernel")

        assert input_dim == 1

        self.basis_func = basis_func

        self.a = GPy.core.parameterization.Param("a", a)
        self.b = GPy.core.parameterization.Param("b", b)

        self.link_parameters(self.a, self.b)

    def K(self, X, X2):
        if X2 is None: X2 = X

        X_ = self.basis_func(X)
        X2_ = self.basis_func(X2)
        k = np.dot(X_ * self.b, X2_.T) + self.a

        return k

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        X_ = self.basis_func(X)
        X2_ = self.basis_func(X2)
        self.a.gradient = np.sum(dL_dK)
        self.b.gradient = np.sum(np.dot(np.dot(X_, X2_.T), dL_dK))

    def Kdiag(self, X):
        return np.diag(self.K(X, X))


def linear(s):
    return s


def quad(s):
    return (1 - s) ** 2


def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform


def retransform(s_transform, s_min, s_max):
    s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return s


class FabolasModel(GPyModelWrapper):

    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray,
                 s_min: float, s_max: float, basis_func=linear, noise: float = 1e-6):
        """
        Fabolas Gaussian processes model which models the validation error / cost of
        hyperparameter configurations across training dataset subsets.

        :param X_init: training data points
        :param Y_init: training targets
        :param basis_func: basis function which describes the change in performance across dataset subsets
        :param noise: observation noise added to the diagonal of the kernel matrix
        """

        self.noise = noise
        self.s_min = s_min
        self.s_max = s_max
        self._X = deepcopy(X_init)
        self._X[:, -1] = transform(self._X[:, -1], self.s_min, self.s_max)
        self._Y = Y_init
        self.basis_func = basis_func
        kernel = GPy.kern.Matern52(input_dim=self._X.shape[1] - 1, active_dims=[i for i in range(self._X.shape[1] - 1)],
                                   variance=np.var(self._Y), ARD=True)
        kernel *= FabolasKernel(input_dim=1, active_dims=[self._X.shape[1] - 1], basis_func=basis_func)
        kernel += GPy.kern.White(input_dim=1, active_dims=[self._X.shape[1] - 1], variance=1e-6)

        gp = GPy.models.GPRegression(self._X, self._Y, kernel=kernel, noise_var=noise)
        gp.kern.set_prior(GPy.priors.Uniform(0, 5))
        gp.likelihood.constrain_positive()
        super(FabolasModel, self).__init__(gpy_model=gp, n_restarts=3)

    def predict(self, X):
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        X_ = deepcopy(X)
        X_[:, -1] = transform(X_[:, -1], self.s_min, self.s_max)
        return super(FabolasModel, self).predict(X_)

    def set_data(self, X, Y):
        """
        Sets training data in model

        :param X: New training features
        :param Y: New training outputs
        """
        self._X = deepcopy(X)
        self._X[:, -1] = transform(self._X[:, -1], self.s_min, self.s_max)
        self._Y = Y
        try:
            self.model.set_XY(self._X, self.Y)
        except:
            kernel = GPy.kern.Matern52(input_dim=self._X.shape[1] - 1,
                                       active_dims=[i for i in range(self._X.shape[1] - 1)],
                                       variance=np.var(self.Y), ARD=True)
            kernel *= FabolasKernel(input_dim=1, active_dims=[self._X.shape[1] - 1], basis_func=self.basis_func)
            kernel *= GPy.kern.OU(input_dim=1, active_dims=[self._X.shape[1] - 1])

            self.model = GPy.models.GPRegression(self._X, self.Y, kernel=kernel, noise_var=self.noise)
            self.model.likelihood.constrain_positive()

    def get_f_minimum(self):
        """
        Predicts for all observed data points the validation error on the full dataset and returns
        the smallest mean prediciton

        :return: Array of size 1 x 1
        """
        proj_X = deepcopy(self._X)
        proj_X[:, -1] = np.ones(proj_X.shape[0]) * self.s_max
        mean_highest_dataset = self.model.predict(proj_X)

        return np.min(mean_highest_dataset, axis=0)

    @property
    def X(self):
        X = deepcopy(self._X)
        X[:, -1] = retransform(X[:, -1], self.s_min, self.s_max)
        return X

    @property
    def Y(self):
        return self._Y

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        X_ = deepcopy(X)
        X_[:, -1] = transform(X_[:, -1], self.s_min, self.s_max)

        return super(FabolasModel, self).get_prediction_gradients(X_)

    def predict_covariance(self, X: np.ndarray, with_noise: bool = True) -> np.ndarray:
        """
        Calculates posterior covariance between points in X
        :param X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
        :param with_noise: Whether to include likelihood noise in the covariance matrix
        :return: Posterior covariance matrix of size n_points x n_points
        """
        X_ = deepcopy(X)
        X_[:, -1] = transform(X_[:, -1], self.s_min, self.s_max)

        return super(FabolasModel, self).predict_covariance(X_, with_noise)

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate posterior covariance between two points
        :param X1: An array of shape 1 x n_dimensions that contains a data single point. It is the first argument of the
                   posterior covariance function
        :param X2: An array of shape n_points x n_dimensions that may contain multiple data points. This is the second
                   argument to the posterior covariance function.
        :return: An array of shape n_points x 1 of posterior covariances between X1 and X2
        """
        X_1 = deepcopy(X1)
        X_1[:, -1] = transform(X_1[:, -1], self.s_min, self.s_max)
        X_2 = deepcopy(X2)
        X_2[:, -1] = transform(X_2[:, -1], self.s_min, self.s_max)

        return super(FabolasModel, self).get_covariance_between_points(X1, X2)
