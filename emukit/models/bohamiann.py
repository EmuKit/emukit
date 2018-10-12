# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ..core.interfaces.models import IModel, IDifferentiable

try:
    from pybnn import bohamiann
except ImportError:
    raise ImportError("""
        This module is missing required dependencies. Try running

        pip install git+https://github.com/automl/pybnn.git
    """)


class Bohamiann(IModel, IDifferentiable):

    def __init__(self, X_init, Y_init, **kwargs):
        """
        Implements Bayesian neural networks as described by Springenberg et. al[1] based on
        stochastic gradient Hamiltonian monte carlo sampling[2].

        Dependencies:
            AutoML pybnn (https://github.com/automl/pybnn)

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            Bayesian Optimization with Robust Bayesian Neural Networks.
            In Advances in Neural Information Processing Systems 29 (2016).
        [2] T. Chen, E. B. Fox, C. Guestrin
            Stochastic Gradient Hamiltonian Monte Carlo
            Proceedings of the 31st International Conference on Machine Learning
        """
        super().__init__()

        self.model = bohamiann.Bohamiann()

        self._X = X_init
        self._Y = Y_init

        self.model.train(X_init, Y_init, **kwargs)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for given points

        :param X: points to run prediction for
        """
        m, v = self.model.predict(X)

        return m[:, None], v[:, None]

    def update_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Updates model with new data points.

        :param X: new points
        :param Y: function values at new points X
        """
        self._X = np.append(self._X, X, axis=0)
        self._Y = np.append(self._Y, Y, axis=0)

        self.model.train(self._X, self._Y)

    def optimize(self) -> None:
        pass

    def get_f_minimum(self):
        return np.min(self._Y)

    def get_prediction_gradients(self, X: np.ndarray) -> np.ndarray:
        """
        Computes and returns model gradients at given points

        :param X: points to compute gradients at
        """

        dm = np.array([self.model.predictive_mean_gradient(xi) for xi in X])
        dv = np.array([self.model.predictive_variance_gradient(xi) for xi in X])

        return dm, dv
