# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Optional

import numpy as np
import GPy

from ..core.interfaces import IModel, IDifferentiable, IJointlyDifferentiable, IPriorHyperparameters, IModelWithNoise
from ..experimental_design.interfaces import ICalculateVarianceReduction
from ..bayesian_optimization.interfaces import IEntropySearchModel


class GPyModelWrapper(
    IModel, IDifferentiable, IJointlyDifferentiable, ICalculateVarianceReduction, IEntropySearchModel, IPriorHyperparameters, IModelWithNoise
):
    """
    This is a thin wrapper around GPy models to allow users to plug GPy models into Emukit
    """
    def __init__(self, gpy_model: GPy.core.Model, n_restarts: int = 1):
        """
        :param gpy_model: GPy model object to wrap
        :param n_restarts: Number of restarts during hyper-parameter optimization
        """
        self.model = gpy_model
        self.n_restarts = n_restarts

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        return self.model.predict(X)

    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        return self.model.predict(X, include_likelihood=False)

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 and n_points x n_points of the predictive
                 mean and variance at each input location
        """
        return self.model.predict(X, full_cov=True)

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        d_mean_dx, d_variance_dx = self.model.predictive_gradients(X)
        return d_mean_dx[:, :, 0], d_variance_dx

    def get_joint_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes and returns model gradients of mean and full covariance matrix at given points

        :param X: points to compute gradients at, nd array of shape (q, d)
        :return: Tuple with first item being gradient of the mean of shape (q) at X with respect to X (return shape is (q, q, d)).
                 The second item is the gradient of the full covariance matrix of shape (q, q) at X with respect to X
                 (return shape is (q, q, q, d)).
        """
        dmean_dx = dmean(X, self.model.X, self.model.kern, self.model.posterior.woodbury_vector[:, 0])
        dvariance_dx = dSigma(X, self.model.X, self.model.kern, self.model.posterior.woodbury_inv)
        return dmean_dx, dvariance_dx

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: New training features
        :param Y: New training outputs
        """
        self.model.set_XY(X, Y)

    def optimize(self, verbose=False):
        """
        Optimizes model hyper-parameters
        """
        self.model.optimize_restarts(self.n_restarts, verbose=verbose, robust=True)

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """
        Computes the variance reduction at x_test, if a new point at x_train_new is acquired
        """
        covariance = self.model.posterior_covariance_between_points(x_train_new, x_test, include_likelihood=False)
        variance_prediction = self.model.predict(x_train_new)[1]
        return covariance**2 / variance_prediction

    def predict_covariance(self, X: np.ndarray, with_noise: bool=True) -> np.ndarray:
        """
        Calculates posterior covariance between points in X
        :param X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
        :param with_noise: Whether to include likelihood noise in the covariance matrix
        :return: Posterior covariance matrix of size n_points x n_points
        """
        _, v = self.model.predict(X, full_cov=True, include_likelihood=with_noise)
        v = np.clip(v, 1e-10, np.inf)

        return v

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate posterior covariance between two sets of points.
        :param X1: An array of shape n_points1 x n_dimensions. This is the first argument of the
                   posterior covariance function.
        :param X2: An array of shape n_points2 x n_dimensions. This is the second argument of the
                   posterior covariance function.
        :return: An array of shape n_points1 x n_points2 of posterior covariances between X1 and X2.
            Namely, [i, j]-th entry of the returned array will represent the posterior covariance
            between i-th point in X1 and j-th point in X2.
        """
        return self.model.posterior_covariance_between_points(X1, X2, include_likelihood=False)

    @property
    def X(self) -> np.ndarray:
        """
        :return: An array of shape n_points x n_dimensions containing training inputs
        """
        return self.model.X

    @property
    def Y(self) -> np.ndarray:
        """
        :return: An array of shape n_points x 1 containing training outputs
        """
        return self.model.Y

    def generate_hyperparameters_samples(self, n_samples=20, n_burnin=100, subsample_interval=10,
                                         step_size=1e-1, leapfrog_steps=20) -> np.ndarray:
        """
        Generates the samples from the hyper-parameters and returns them.
        :param n_samples: Number of generated samples.
        :param n_burnin: Number of initial samples not used.
        :param subsample_interval: Interval of subsampling from HMC samples.
        :param step_size: Size of the gradient steps in the HMC sampler.
        :param leapfrog_steps: Number of gradient steps before each Metropolis Hasting step.
        :return: A numpy array whose rows are samples of the hyper-parameters.

        """
        self.model.optimize(max_iters=self.n_restarts)
        # Add jitter to all unfixed parameters. After optimizing the hyperparameters, the gradient of the
        # posterior probability of the parameters wrt. the parameters will be close to 0.0, which is a poor
        # initialization for HMC
        unfixed_params = [param for param in self.model.flattened_parameters if not param.is_fixed]
        for param in unfixed_params:
            # Add jitter by multiplying with log-normal noise with mean 1 and standard deviation 0.01 
            # This ensures the sign of the parameter remains the same
            param *= np.random.lognormal(np.log(1. / np.sqrt(1.0001)), np.sqrt(np.log(1.0001)), size=param.size)
        hmc = GPy.inference.mcmc.HMC(self.model, stepsize=step_size)
        samples = hmc.sample(num_samples=n_burnin + n_samples * subsample_interval, hmc_iters=leapfrog_steps)
        return samples[n_burnin::subsample_interval]

    def fix_model_hyperparameters(self, sample_hyperparameters: np.ndarray) -> None:
        """
        Fix model hyperparameters

        """
        if self.model._fixes_ is None:
            self.model[:] = sample_hyperparameters
        else:
            self.model[self.model._fixes_] = sample_hyperparameters
        self.model._trigger_params_changed()


def dSigma(x_predict: np.ndarray, x_train: np.ndarray, kern: GPy.kern, w_inv: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the posterior covariance with respect to the prediction input

    :param x_predict: Prediction inputs of shape (q, d)
    :param x_train: Training inputs of shape (n, d)
    :param kern: Covariance of the GP model
    :param w_inv: Woodbury inverse of the posterior fit of the GP
    :return: Gradient of the posterior covariance of shape (q, q, q, d)
    """
    q, d, n = x_predict.shape[0], x_predict.shape[1], x_train.shape[0]
    dkxX_dx = np.empty((q, n, d))
    dkxx_dx = np.empty((q, q, d))
    for i in range(d):
        dkxX_dx[:, :, i] = kern.dK_dX(x_predict, x_train, i)
        dkxx_dx[:, :, i] = kern.dK_dX(x_predict, x_predict, i)
    K = kern.K(x_predict, x_train)

    dsigma = np.zeros((q, q, q, d))
    for i in range(q):
        for j in range(d):
            Ks = np.zeros((q, n))
            Ks[i, :] = dkxX_dx[i, :, j]
            dKss_dxi = np.zeros((q, q))
            dKss_dxi[i, :] = dkxx_dx[i, :, j]
            dKss_dxi[:, i] = dkxx_dx[i, :, j].T
            dKss_dxi[i, i] = 0
            dsigma[:, :, i, j] = dKss_dxi - Ks @ w_inv @ K.T - K @ w_inv @ Ks.T
    return dsigma


def dmean(x_predict: np.ndarray, x_train: np.ndarray, kern: GPy.kern, w_vec: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the posterior mean with respect to prediction input

    :param x_predict: Prediction inputs of shape (q, d)
    :param x_train: Training inputs of shape (n, d)
    :param kern: Covariance of the GP model
    :param w_vec: Woodbury vector of the posterior fit of the GP
    :return: Gradient of the posterior mean of shape (q, q, d)
    """
    q, d, n = x_predict.shape[0], x_predict.shape[1], x_train.shape[0]
    dkxX_dx = np.empty((q, n, d))
    dmu = np.zeros((q, q, d))
    for i in range(d):
        dkxX_dx[:, :, i] = kern.dK_dX(x_predict, x_train, i)
        for j in range(q):
            dmu[j, j, i] = (dkxX_dx[j, :, i][None, :] @ w_vec[:, None]).flatten()
    return dmu


class GPyMultiOutputWrapper(IModel, IDifferentiable, ICalculateVarianceReduction, IEntropySearchModel):
    """
    A wrapper around GPy multi-output models.
    X inputs should have the corresponding output index as the last column in the array
    """

    def __init__(self, gpy_model: GPy.core.GP, n_outputs: int, n_optimization_restarts: int,
                 verbose_optimization: bool=True):
        """
        :param gpy_model: GPy multi-output model
        :param n_outputs: Number of outputs in the problem
        :param n_optimization_restarts: Number of restarts from random starting points when optimizing hyper-parameters
        """
        super().__init__()
        self.gpy_model = gpy_model
        self.n_optimization_restarts = n_optimization_restarts
        self.n_outputs = n_outputs
        self.verbose_optimization = verbose_optimization
        self.samples: Optional[np.ndarray] = None

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """
        Calculates reduction in variance at x_test due to observing training point x_train_new

        :param x_train_new: New training point
        :param x_test: Test points to calculate variance reduction at
        :return: Array of variance reduction at each test point
        """
        fidelities_train_new = x_train_new[:, -1]
        y_metadata = {'output_index': fidelities_train_new.astype(int)}
        covariance = self.gpy_model.posterior_covariance_between_points(x_train_new, x_test, include_likelihood=False)
        variance_prediction = self.gpy_model.predict(x_train_new, Y_metadata=y_metadata)[1]
        return covariance**2 / variance_prediction

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates gradients of predictions with respect to X, excluding with respect to the output index
        :param X: Point at which to predict gradients
        :return: (mean gradient, variance gradient)
        """
        dmean_dx, dvar_dx = self.gpy_model.predictive_gradients(X)
        return dmean_dx[:, :-1], dvar_dx[:, :-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts mean and variance for output specified by last column of X
        :param X: point(s) at which to predict
        :return: predicted (mean, variance) at X
        """
        output_index = X[:, -1]
        y_metadata = {'output_index': output_index.astype(int)}
        return self.gpy_model.predict(X, Y_metadata=y_metadata)

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Updates model with new training data
        :param X: New training features with output index as last column
        :param Y: New training targets with output index as last column
        """
        y_metadata = {'output_index': X[:, -1].astype(int)}
        self.gpy_model.update_model(False)
        self.gpy_model.Y_metadata = y_metadata
        self.gpy_model.set_XY(X, Y)
        self.gpy_model.update_model(True)

    def optimize(self) -> None:
        """
        Optimizes hyper-parameters of model. Starts the optimization at random locations equal to the values of the
        "n_optimization_restarts" attribute.
        """
        # Optimize the model if optimization_restarts > 0
        if self.n_optimization_restarts == 1:
            self.gpy_model.optimize()
        elif self.n_optimization_restarts >= 1:
            self.gpy_model.optimize_restarts(self.n_optimization_restarts, verbose=self.verbose_optimization,
                                             robust=True)

    @property
    def X(self) -> np.ndarray:
        return self.gpy_model.X

    @property
    def Y(self) -> np.ndarray:
        return self.gpy_model.Y

    def predict_covariance(self, X: np.ndarray, with_noise: bool = True) -> np.ndarray:
        """
        Calculates posterior covariance between points in X

        :param X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
        :param with_noise: Whether to include likelihood noise in the covariance matrix
        :return: Posterior covariance matrix of size n_points x n_points
        """
        output_index = X[:, -1]
        y_metadata = {'output_index': output_index.astype(int)}
        variance = self.gpy_model.predict(X, Y_metadata=y_metadata, full_cov=True)[1]
        variance = np.maximum(variance, 1e-10)
        return variance

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate posterior covariance between two points
        :param X1: An array of shape 1 x n_dimensions that contains a data single point. It is the first argument of the
                   posterior covariance function
        :param X2: An array of shape n_points x n_dimensions that may contain multiple data points. This is the second
                   argument to the posterior covariance function.
        :return: An array of shape n_points x 1 of posterior covariances between X1 and X2
        """
        return self.gpy_model.posterior_covariance_between_points(X1, X2, include_likelihood=False)

    def generate_hyperparameters_samples(self, n_samples = 10, n_burnin = 5, subsample_interval = 1,
                                         step_size = 1e-1, leapfrog_steps = 1) -> np.ndarray:
        """
        Generates the samples from the hyper-parameters, and returns them (a numpy array whose rows are
        samples of the hyper-parameters).
        :param n_samples: Number of generated samples.
        :param n_burnin: Number of initial samples not used.
        :param subsample_interval: Interval of subsampling from HMC samples.
        :param step_size: Size of the gradient steps in the HMC sampler.
        :param leapfrog_steps: Number of gradient steps before each Metropolis Hasting step.
        """
        self.gpy_model.optimize(max_iters=self.n_optimization_restarts)
        self.gpy_model.param_array[:] = self.gpy_model.param_array * (1.+np.random.randn(self.gpy_model.param_array.size)*0.01)
        hmc = GPy.inference.mcmc.HMC(self.gpy_model, stepsize = step_size)
        samples = hmc.sample(num_samples = n_burnin + n_samples * subsample_interval, hmc_iters = leapfrog_steps)
        return samples[n_burnin::subsample_interval]

    def fix_model_hyperparameters(self, sample_hyperparameters: np.ndarray) -> None:
        """
        Fix model hyperparameters

        """
        if self.gpy_model._fixes_ is None:
            self.gpy_model[:] = sample_hyperparameters
        else:
            self.gpy_model[self.gpy_model._fixes_] = sample_hyperparameters
        self.gpy_model._trigger_params_changed()
