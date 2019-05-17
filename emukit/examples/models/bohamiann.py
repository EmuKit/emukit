# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...core.interfaces.models import IModel, IDifferentiable

try:
    from pybnn import bohamiann
except ImportError:
    raise ImportError("""
        This module is missing required dependencies. Try running

        pip install git+https://github.com/automl/pybnn.git

        Refer to https://github.com/automl/pybnn for further information.
    """)

import torch
import torch.nn as nn


def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = nn.Parameter(torch.Tensor(1, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            nn.init.constant_(module.bias, val=np.log(1e-3))
        elif type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(module.bias, val=0.0)

    return nn.Sequential(
        nn.Linear(input_dimensionality, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)


class Bohamiann(IModel, IDifferentiable):

    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray, num_steps: int = 5000, num_burnin: int = 5000,
                 lr: float = 1e-2, get_architecture=get_default_network,
                 **kwargs) -> None:
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

        :param X_init: training data points
        :param Y_init: training function values
        :param num_steps: number of MCMC steps (after the burnin)
        :param num_burnin: number of burnin steps
        :param lr: learning rate or step length of the MCMC sampler
        :param get_architecture: function handle that returns an architecture for the Bayesian neural network
        """
        super().__init__()

        self.model = bohamiann.Bohamiann(get_network=get_architecture)
        self.num_steps = num_steps
        self.num_burnin = num_burnin
        self._X = X_init
        self._Y = Y_init

        self.model.train(X_init, Y_init, num_steps=self.num_steps + self.num_burnin, lr=lr,
                         num_burn_in_steps=self.num_burnin, keep_every=100, **kwargs)

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

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: function values at new points X
        """
        self._X = X
        self._Y = Y

        self.model.train(self._X, self._Y, num_steps=self.num_steps,
                         num_burn_in_steps=self.num_burnin, keep_every=100, verbose=True)

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
