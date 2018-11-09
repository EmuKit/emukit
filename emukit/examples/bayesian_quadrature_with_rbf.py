
from typing import Tuple

import numpy as np

from GPy.kern import RBF
from GPy.models import GPRegression

from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.core import MultiDimensionalContinuousParameter
from emukit.model_wrappers import gpy_wrapper_for_quadrature



class BayesianQuadratureWithRBFKernel(VanillaBayesianQuadrature):
    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 integral_bounds: MultiDimensionalContinuousParameter, kernel_hyperparams: Tuple,
                 noiseless: bool = False):
        """
        Class for Vanilla Bayesian Quadrature with a GPy model at the backend

        :param X: initial input values where the objective has been evaluated.
        :param Y: initial output values where the objective has been evaluated.
        :param integral_bounds: Integration bounds
        :param kernel_hyperparams: Hyperparameters of the RBF kernel (lengthscale, outputscale)
        """
        self._X = X
        self._Y = Y
        self.integral_bounds = integral_bounds
        self.k_hyperparams = kernel_hyperparams
        self.noiseless = noiseless

        # set up an emukit.quadrature.base_gp model
        self._set_up_model()

        super(BayesianQuadratureWithRBFKernel, self).__init__(self.base_gp)


    def _set_up_model(self):
        """ creates an base_gp model for emukit.quadrature from a gpy_model """
        input_dim = self._X.shape[1]

        gpy_kern = RBF(input_dim=input_dim, lengthscale=self.k_hyperparams[0], variance=self.k_hyperparams[1])
        gpmodel = GPRegression(X=self._X, Y=self._Y, kernel=gpy_kern)

        if self.noiseless:
            gpmodel.Gaussian_noise.constrain_fixed(1.e-10)

        self.base_gp = gpy_wrapper_for_quadrature(gpy_model=gpmodel, integral_bounds=self.integral_bounds)


