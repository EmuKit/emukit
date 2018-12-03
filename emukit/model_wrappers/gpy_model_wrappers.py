# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np
import GPy

from ..core.interfaces import IModel, IDifferentiable
from ..experimental_design.interfaces import ICalculateVarianceReduction
from ..bayesian_optimization.interfaces import IEntropySearchModel


class GPyModelWrapper(IModel, IDifferentiable, ICalculateVarianceReduction, IEntropySearchModel):
    """
    This is a thin wrapper around GPy models to allow users to plug GPy models into Emukit
    """
    def __init__(self, gpy_model):
        self.model = gpy_model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        return self.model.predict(X)

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        d_mean_dx, d_variance_dx = self.model.predictive_gradients(X)
        return d_mean_dx[:, :, 0], d_variance_dx

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: New training features
        :param Y: New training outputs
        """
        self.model.set_XY(X, Y)

    def optimize(self):
        """
        Optimizes model hyper-parameters
        """
        self.model.optimize()

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """
        Computes the variance reduction at x_test, if a new point at x_train_new is acquired
        """
        covariance = self.model.posterior_covariance_between_points(x_train_new, x_test)
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
        Calculate posterior covariance between two points
        :param X1: An array of shape 1 x n_dimensions that contains a data single point. It is the first argument of the
                   posterior covariance function
        :param X2: An array of shape n_points x n_dimensions that may contain multiple data points. This is the second
                   argument to the posterior covariance function.
        :return: An array of shape n_points x 1 of posterior covariances between X1 and X2
        """
        return self.model.posterior_covariance_between_points(X1, X2)

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

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """
        Calculates reduction in variance at x_test due to observing training point x_train_new

        :param x_train_new: New training point
        :param x_test: Test points to calculate variance reduction at
        :return: Array of variance reduction at each test point
        """
        fidelities_train_new = x_train_new[:, -1]
        y_metadata = {'output_index': fidelities_train_new.astype(int)}
        covariance = self.gpy_model.posterior_covariance_between_points(x_train_new, x_test)
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
            self.gpy_model.optimize_restarts(self.n_optimization_restarts, verbose=self.verbose_optimization)

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
        return self.gpy_model.posterior_covariance_between_points(X1, X2)
