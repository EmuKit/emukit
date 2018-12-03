# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


class IEntropySearchModel:
    """
    Interface containing abstract methods that need to be implemented if using entropy search Bayesian optimization
    acquisition function.
    """
    def predict_covariance(self, X: np.ndarray, with_noise: bool=True) -> np.ndarray:
        """

        :param X: Numpy array of shape (n_points, n_features) of test input locations
        :param with_noise: Whether to include likelihood noise term in covariance
        :return: Posterior covariance matrix which has shape (n_points x n_points)
        """
        raise NotImplementedError

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate posterior covariance between one point in X1 and all points in X2

        :param X1: An array of shape 1 x n_dimensions that contains a data single point. It is the first argument of the
                   posterior covariance function
        :param X2: An array of shape n_points x n_dimensions that may contain multiple data points. This is the second
                   argument to the posterior covariance function.
        :return: An array of shape n_points x 1 of posterior covariances between X1 and X2
        """
        raise NotImplementedError
