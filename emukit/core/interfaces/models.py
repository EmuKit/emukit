# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple


class IModel:
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: function values at new points X
        """
        raise NotImplementedError

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        raise NotImplementedError

    @property
    def X(self):
        raise NotImplementedError

    @property
    def Y(self):
        raise NotImplementedError


class IDifferentiable:
    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """
        Computes and returns model gradients of mean and variance at given points

        :param X: points to compute gradients at
        :returns: Tuple of gradients of mean and variance.
        """
        raise NotImplementedError


class IJointlyDifferentiable:
    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 and n_points x n_points of the predictive
                 mean and variance at each input location
        """
        raise NotImplementedError

    def get_joint_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes and returns model gradients of mean and full covariance matrix at given points

        :param X: points to compute gradients at, nd array of shape (q, d)
        :return: Tuple with first item being gradient of the mean of shape (q) at X with respect to X (return shape is (q, q, d)).
                 The second item is the gradient of the full covariance matrix of shape (q, q) at X with respect to X
                 (return shape is (q, q, q, d)).
        """
        raise NotImplementedError


class ICrossCovarianceDifferentiable:
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
        raise NotImplementedError

    def get_covariance_between_points_gradients(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the posterior covariance matrix between prediction at inputs x1 and x2
        with respect to x1.

        :param X1: Prediction inputs of shape (q1, d)
        :param X2: Prediction inputs of shape (q2, d)
        :return: nd array of shape (q1, q2, d) representing the gradient of the posterior covariance
            between x1 and x2 with respect to x1. res[i, j, k] is the gradient of Cov(y1[i], y2[j])
            with respect to x1[i, k]
        """
        raise NotImplementedError


class IPriorHyperparameters:
    def generate_hyperparameters_samples(self, n_samples: int, n_burnin: int,
                                         subsample_interval: int, step_size: float, leapfrog_steps: int) -> np.ndarray:
        """
        Generates the samples from the hyper-parameters of the model, and returns them.

        :param n_samples: Number of hyper-parameter samples
        :param n_burnin: Number of initial samples not used.
        :param subsample_interval: Interval of subsampling from HMC samples.
        :param step_size: Size of the gradient steps in the HMC sampler.
        :param leapfrog_steps: Number of gradient steps before each Metropolis Hasting step.
        """
        raise NotImplementedError

    def fix_model_hyperparameters(self, sample_hyperparameters: np.ndarray) -> None:
        """
        Fixes the model hyper-parameters to certain values (which can be taken from samples).

        :param sample_hyperparameters: np.ndarray whose rows contain each hyper-parameters set.
        """
        raise NotImplementedError


class IModelWithNoise:
    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For given points X, predict mean and variance of the output without observation noise.

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError
