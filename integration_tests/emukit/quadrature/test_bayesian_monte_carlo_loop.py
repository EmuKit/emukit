# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import GPy
import pytest

from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.quadrature.loop.bayesian_monte_carlo_loop import BayesianMonteCarlo
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.model_wrappers.gpy_quadrature_wrappers import QuadratureRBFLebesgueMeasure, RBFGPy, BaseGaussianProcessGPy

from numpy.testing import assert_array_equal


def func(x):
    return np.ones((x.shape[0], 1))


@pytest.fixture
def loop():
    init_size = 5
    x_init = np.random.rand(init_size, 2)
    y_init = np.random.rand(init_size, 1)
    bounds = [(-1, 1), (0, 1)]

    gpy_model = GPy.models.GPRegression(X=x_init, Y=y_init, kernel=GPy.kern.RBF(input_dim=x_init.shape[1],
                                                                                lengthscale=1., variance=1.))
    emukit_qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), integral_bounds=bounds)
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=x_init, Y=y_init)
    emukit_loop = BayesianMonteCarlo(model=emukit_method)
    return emukit_loop, init_size, x_init, y_init


def test_bayesian_monte_carlo_loop(loop):
    emukit_loop, init_size, _, _ = loop

    num_iter = 5
    emukit_loop.run_loop(user_function=UserFunctionWrapper(func), stopping_condition=num_iter)

    assert emukit_loop.loop_state.X.shape[0] == num_iter + init_size
    assert emukit_loop.loop_state.Y.shape[0] == num_iter + init_size


def test_bayesian_monte_carlo_loop_initial_state(loop):
    emukit_loop, _, x_init, y_init = loop

    assert_array_equal(emukit_loop.loop_state.X, x_init)
    assert_array_equal(emukit_loop.loop_state.Y, y_init)
    assert emukit_loop.loop_state.iteration == 0
