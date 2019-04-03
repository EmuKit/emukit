"""
These are emukit model wrappers that contain the specific optimization procedures we found worked well for each model.

The constructor for each model takes X and Y as lists, with each entry of the list corresponding to data for a fidelity
"""
import logging

import GPy
import numpy as np

from ...core.interfaces import IModel
from ...model_wrappers import GPyMultiOutputWrapper
from ...multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from ...multi_fidelity.kernels import LinearMultiFidelityKernel
from ...multi_fidelity.models import GPyLinearMultiFidelityModel
from ...multi_fidelity.models.non_linear_multi_fidelity_model import (
    NonLinearMultiFidelityModel, make_non_linear_kernels)

_log = logging.getLogger(__name__)


class HighFidelityGp(IModel):
    """
    GP at high fidelity only.

    The optimization is restarted from random initial points 10 times.
    The noise parameter is initialized at 1e-6 for the first optimization round.
    """

    def __init__(self, X, Y):
        kern = GPy.kern.RBF(X[1].shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X[1], Y[1], kernel=kern)
        self.model.Gaussian_noise.variance = 1e-6
        self.name = 'hf_gp'

    def optimize(self):
        _log.info('\n--- Optimization: ---\n'.format(self.name))
        self.model.optimize_restarts(10, robust=True)

    def predict(self, X):
        """
        Predict from high fidelity
        """
        return self.model.predict(X[:, :-1])

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError()

    @property
    def X(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()


class LinearAutoRegressiveModel(IModel):
    """
    Linear model, AR1 in paper. Optimized with noise fixed at 1e-6 until convergence then the noise parameter is freed
    and the model is optimized again
    """
    def __init__(self, X, Y, n_restarts=10):
        """

        :param X: List of training data at each fidelity
        :param Y: List of training targets at each fidelity
        :param n_restarts: Number of restarts during optimization of hyper-parameters
        """
        x_train, y_train = convert_xy_lists_to_arrays(X, Y)
        n_dims = X[0].shape[1]
        kernels = [GPy.kern.RBF(n_dims, ARD=True) for _ in range(len(X))]
        lin_mf_kernel = LinearMultiFidelityKernel(kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(x_train, y_train, lin_mf_kernel, n_fidelities=len(X))
        gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(1e-6)
        gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(1e-6)
        if len(Y) == 3:
            gpy_lin_mf_model.mixed_noise.Gaussian_noise_2.fix(1e-6)

        self.model = GPyMultiOutputWrapper(gpy_lin_mf_model, len(X), n_optimization_restarts=n_restarts)
        self.name = 'ar1'
        self.n_fidelities = len(X)

    def predict(self, X):
        """
        Predict from high fidelity
        """
        return self.model.predict(X)

    def optimize(self):
        _log.info('\n--- Optimization: ---\n'.format(self.name))
        self.model.optimize()
        self.model.gpy_model.mixed_noise.Gaussian_noise.unfix()
        self.model.gpy_model.mixed_noise.Gaussian_noise_1.unfix()
        if self.n_fidelities == 3:
            self.model.gpy_model.mixed_noise.Gaussian_noise_2.unfix()
        self.model.optimize()

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError()

    @property
    def X(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()


class NonLinearAutoRegressiveModel(IModel):
    """
    Non-linear model, NARGP in paper
    """
    def __init__(self, X, Y, n_restarts=10):
        x_train, y_train = convert_xy_lists_to_arrays(X, Y)
        base_kernel = GPy.kern.RBF
        kernels = make_non_linear_kernels(base_kernel, len(X), x_train.shape[1] - 1, ARD=True)
        self.model = NonLinearMultiFidelityModel(x_train, y_train, n_fidelities=len(X), kernels=kernels,
                                                 verbose=True, optimization_restarts=n_restarts)
        self.name = 'nargp'

    def predict(self, X):
        """
        Predict from high fidelity
        """
        return self.model.predict(X)

    def optimize(self):
        _log.info('\n--- Optimization: ---\n'.format(self.name))
        self.model.optimize()

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError()

    @property
    def X(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()
