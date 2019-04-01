"""
These are emukit model wrappers that contain the specific optimization procedures we found worked well for each model.

The constructor for each model takes X and Y as lists, with each entry of the list corresponding to the
"""

import GPy
import numpy as np

import emukit
from emukit.core.interfaces import IModel
from emukit.model_wrappers import GPyMultiOutputWrapper, GPyModelWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import (
    NonLinearMultiFidelityModel, make_non_linear_kernels)


class HfGpOnly(IModel):
    """
    GP at high fidelity only
    """

    def __init__(self, X, Y):
        kern = GPy.kern.RBF(X[1].shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X[1], Y[1], kernel=kern)
        self.model.Gaussian_noise.variance = 1e-6
        self.name = 'hf_gp'

    def optimize(self):
        print('\n--- Optimization: ',self.name,' ---\n')
        self.model.optimize_restarts(10, robust=True)

    def predict(self, X):
        return self.model.predict(X[:, :-1])

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError()

    @property
    def X(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()


class Ar1Model(IModel):
    """
    Linear model. Optimized with noise fixed at 1e-6 then this is freed and optimization continues
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
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(x_train, y_train, lin_mf_kernel, n_fidelities=len(X))
        gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(1e-6)
        gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(1e-6)
        if len(Y) == 3:
            gpy_lin_mf_model.mixed_noise.Gaussian_noise_2.fix(1e-6)

        self.model = GPyMultiOutputWrapper(gpy_lin_mf_model, len(X), n_optimization_restarts=n_restarts)
        self.name = 'ar1'
        self.n_fidelities = len(X)

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self):
        print('\n--- Optimization: ',self.name,' ---\n')
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


class Nargp(IModel):
    """
    Non-linear model.
    """
    def __init__(self, X, Y, n_restarts=10):
        x_train, y_train = convert_xy_lists_to_arrays(X, Y)
        base_kernel = GPy.kern.RBF
        kernels = make_non_linear_kernels(base_kernel, len(X), x_train.shape[1] - 1)#, ARD=True)
        self.model = NonLinearMultiFidelityModel(x_train, y_train, n_fidelities=len(X), kernels=kernels,
                                                 verbose=True, optimization_restarts=n_restarts)
        self.name = 'nargp'

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self):
        print('\n--- Optimization: ',self.name,' ---\n')
        self.model.optimize()

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError()

    @property
    def X(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()
